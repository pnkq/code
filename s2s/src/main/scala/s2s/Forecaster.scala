package s2s

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.{Model, Sequential}
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.keras.metrics.MAE
import com.intel.analytics.bigdl.dllib.keras.models.{KerasNet, Models}
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.MSECriterion
import com.intel.analytics.bigdl.dllib.optim.{Loss, Or, Trigger, MinLoss, MaxEpoch}
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.functions.array_to_vector
import scopt.OptionParser

import java.nio.file.{Files, Paths, StandardOpenOption}

/**
 * An implementation of time series predictor using LSTM and BERT.
 *
 * (C) phuonglh@gmail.com
 *
 */
object Forecaster {

  /**
   * Rolls a data frame based on the "date" column.
   * @param ff a data frame
   * @param lookBack number of past days to look back
   * @param horizon number of future days to forecast
   * @param featureCols column names to row
   * @param targetCol target column
   * @param dayOfYear if given a dayOfYear feature, we roll it (only for the BERT model).
   * @return a data frame with an array of columns for input features
   */
  private def roll(ff: DataFrame, lookBack: Int, horizon: Int, featureCols: Array[String], targetCol: String = "y",
                   dayOfYear: Boolean = false, imputeMissing: Boolean = false): DataFrame = {
    // defining a window ordered by date
    val window = Window.orderBy("date")
    // roll values, each value is in a separate column
    var gf = ff
    // all features and the target value are rolled backward
    // use select() to avoid StackOverflowException
    val gfCols = gf.schema.fieldNames.map(col)
    val lagCols = for {
      name <- featureCols :+ targetCol
      j <- -lookBack + 1 to -1
    } yield {
      lag(name, -j).over(window).as(s"${name}_$j")
    }
    gf = gf.select(gfCols ++ lagCols: _*)
    // the target value is rolled forward
    val allCols = gf.schema.fieldNames.map(col)
    val leadCols = (1 to horizon).map(j => lead(targetCol, j).over(window).as(s"${targetCol}_$j"))
    gf = gf.select(allCols ++ leadCols: _*)
    // roll the dayOfYear featureOr
    if (dayOfYear) {
      val allCols = gf.schema.fieldNames.map(col)
      val lagCols = (-lookBack + 1 to -1).map(j => lag("dayOfYear", -j).over(window).as(s"dayOfYearP${Math.abs(j)}"))
      gf = gf.select(allCols ++ lagCols: _*)
    }
    // impute missing values if specified:
    if (!imputeMissing) gf else {
      // collect all columns that has been rolled for missing value imputation
      val addedCols = gf.schema.fieldNames.filter(c => featureCols.exists(c.indexOf(_) >= 0))
      // fill missing data (null) as zero
      var af = gf
      for (c <- addedCols) {
        af = af.withColumn(c, when(isnull(col(c)), 0).otherwise(col(c)))
      }
      af
    }
  }

  private def createModel(config: Config, featureSize: Int): KerasNet[Float] = {
    config.modelType match {
      case 1 =>
        // LSTM
        val model = Sequential()
        model.add(Reshape(targetShape = Array(config.lookBack, featureSize), inputShape = Shape(config.lookBack * featureSize)).setName("reshape"))
        if (!config.bidirectional) {
          for (j <- 0 until config.nLayer - 1) {
            model.add(LSTM(config.hiddenSize, returnSequences = true).setName(s"lstm-$j"))
          }
        } else {
          for (j <- 0 until config.nLayer - 1) {
            model.add(Bidirectional(LSTM(config.hiddenSize, returnSequences = true).setName(s"lstm-$j")))
          }
        }
        model.add(LSTM(config.hiddenSize).setName(s"lstm-${config.nLayer - 1}"))
        if (config.dropoutRate > 0)
          model.add(Dropout(config.dropoutRate).setName("dropout"))
        model.add(Dense(config.horizon).setName("dense"))
        model
      case 2 =>
        // LSTMs + BERT for dayOfYear
        val input1 = Input(inputShape = Shape(config.lookBack * featureSize))
        val input2 = Input(inputShape = Shape(4*config.lookBack))
        val reshape1 = Reshape(targetShape = Array(config.lookBack, featureSize)).inputs(input1)
        val reshape2 = Reshape(targetShape = Array(4, config.lookBack)).inputs(input2)
        val lstm1 = if (config.bidirectional) {
          Bidirectional(LSTM(config.hiddenSize, returnSequences = true)).inputs(reshape1)
        } else LSTM(config.hiddenSize, returnSequences = true).inputs(reshape1)
        val lstm2 = if (config.bidirectional) {
          Bidirectional(LSTM(config.hiddenSize, returnSequences = false)).inputs(lstm1)
        } else LSTM(config.hiddenSize, returnSequences = false).inputs(lstm1)
        val split = SplitTensor(1, 4).inputs(reshape2)
        val id = SelectTable(0).inputs(split)
        val sId = Squeeze(1).inputs(id)
        val typ = SelectTable(1).inputs(split)
        val sTyp = Squeeze(1).inputs(typ)
        val pos = SelectTable(2).inputs(split)
        val sPos = Squeeze(1).inputs(pos)
        val mask = SelectTable(3).inputs(split)
        val reshapeMask = Reshape(targetShape = Array(1, 1, config.lookBack)).inputs(mask)
        val bert = BERT(hiddenSize = config.bertSize, nBlock = config.nBlock, nHead = config.nHead, maxPositionLen = config.lookBack,
          intermediateSize = config.intermediateSize, outputAllBlock = false).inputs(sId, sTyp, sPos, reshapeMask)
        val poolOutput = SelectTable(1).inputs(bert) // a tensor of shape 1 x hiddenSize
        val merge = Merge.merge(inputs = List(lstm2, poolOutput), mode = "concat")
        val output = Dense(config.horizon).inputs(merge)
        Model(Array(input1, input2), output)
    }
  }


  /**
   * Compute the error between the gold label and the prediction (MAE -- mean absolute error, MSE -- mean squared error)
   * @param df a prediction data frame, which should have two columns: "label" (vector) and "prediction" (array[float])
   */
  private def computeError(df: DataFrame) = {
    val errorMAE = udf((label: Vector, prediction: Array[Float]) => {
      val error = label.toArray.zip(prediction).map { case (a, b) => Math.abs(a - b) }
      Vectors.dense(error)
    })
    val errorMSE = udf((label: Vector, prediction: Array[Float]) => {
      val error = label.toArray.zip(prediction).map { case (a, b) => (a - b)*(a - b) }
      Vectors.dense(error)
    })
    // compute average MAE/MSE between label and prediction
    val output = df.select("label", "prediction")
      .withColumn("mae", errorMAE(col("label"), col("prediction")))
      .withColumn("mse", errorMSE(col("label"), col("prediction")))
    import org.apache.spark.ml.stat.Summarizer
    output.select(Summarizer.mean(col("mae")), Summarizer.mean(col("mse")))
  }

  private def train(ff: DataFrame, config: Config): Result = {
    val dateInputCols = Array("month", "dayOfMonth")
    val targetCol = "y"
    // station-specific data contains "extra" columns
    // region-specific data contains [hgt, relh, uwind, vwind, slp, soilw] columns
    // val extraColumnNames = Set("extra", "hgt", "relh", "wind", "slp", "soilw")
    val extraColumnNames = Set("extra", "_c")
    val extraCols = ff.schema.fieldNames.filter(name => extraColumnNames.exists(e => name.contains(e)))
    println(s"Number of extra features = ${extraCols.length}")
    val featureCols = extraCols ++ dateInputCols // all features features to be rolled

    println(s"Rolling the data for horizon=${config.horizon} and lookBack=${config.lookBack}. Please wait...")
    val af = if (config.modelType == 2) {
      // BERT
      val bf = roll(ff, config.lookBack, config.horizon, featureCols, targetCol, dayOfYear = true)
      bf.withColumn("typeA", array((0 until config.lookBack).map(_ => lit(0)) : _*))
        .withColumn("type", array_to_vector(col("typeA")))
        .withColumn("positionA", array((0 until config.lookBack).map(j => lit(j)) : _*))
        .withColumn("position", array_to_vector(col("positionA")))
        .withColumn("maskA", array((0 until config.lookBack).map(_ => lit(1)) : _*))
        .withColumn("mask", array_to_vector(col("maskA")))
    } else {
      // LSTM (model type = 1)
      roll(ff, config.lookBack, config.horizon, featureCols, targetCol)
    }
    if (config.verbose) {
      println(s"Number of columns of af = ${af.schema.fieldNames.length}")
    }
    // arrange input features by time steps (i.e., -3, -2, -1, 0)
    var inputCols = (-config.lookBack + 1 until 0).toArray.flatMap(j => af.schema.fieldNames.filter(name => name.endsWith(j.toString)))
    inputCols ++= (featureCols ++ Array(targetCol))
    // add rolled cols for dayOfYear feature if using BERT
    if (config.modelType == 2) {
      val dayOfYearCols = af.schema.fieldNames.filter(name => name.contains("dayOfYearP")) :+ "dayOfYear"
      inputCols = inputCols ++ dayOfYearCols
      inputCols = inputCols ++ Array("type", "position", "mask")
    }
    val assemblerX = new VectorAssembler().setInputCols(inputCols).setOutputCol("input").setHandleInvalid("skip")
    val outputCols = (1 to config.horizon).map(c => s"y_$c").toArray
    val assemblerY = new VectorAssembler().setInputCols(outputCols).setOutputCol("output").setHandleInvalid("skip")
    val scalerX = new StandardScaler().setInputCol("input").setOutputCol("features")
    val scalerY = new StandardScaler().setInputCol("output").setOutputCol("label")
    val pipeline = new Pipeline().setStages(Array(assemblerX, assemblerY, scalerX, scalerY))
    val preprocessor = pipeline.fit(af)

    val bf = preprocessor.transform(af)
    // split roughly 90/10, keep sequence order
    val uf = bf.filter("year <= 2015")
    val vf = bf.filter("2016 <= year")
    val featureSize = featureCols.length + 1 // +1 for y (target is also a feature)
    if (config.verbose) {
      println(s"Training size = ${uf.count}")
      println(s"    Test size = ${vf.count}")
      println(s" Feature size = $featureSize")
    }
    val bigdl = createModel(config, featureSize)
    bigdl.summary()
    val trainingSummary = TrainSummary(appName = config.modelType.toString, logDir = s"sum/${config.station}")
    val validationSummary = ValidationSummary(appName = config.modelType.toString, logDir = s"sum/${config.station}")

    val estimator = if (config.modelType == 1) {
      val inputSize = Array(featureSize * config.lookBack)
      NNEstimator[Float](bigdl, new MSECriterion[Float](), featureSize = inputSize, labelSize = Array(config.horizon))
    } else {
      val inputSize = Array(Array(featureSize * config.lookBack), Array(4*config.lookBack))
      NNEstimator[Float](bigdl, new MSECriterion[Float](), featureSize = inputSize, labelSize = Array(config.horizon))
    }
    estimator.setBatchSize(config.batchSize).setOptimMethod(new Adam(lr = config.learningRate))
      .setTrainSummary(trainingSummary).setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, vf, Array(new MAE[Float](), new Loss(new MSECriterion[Float]())), config.batchSize)
//      .setMaxEpoch(config.epochs)
      .setEndWhen(Or(MaxEpoch(config.epochs), MinLoss(config.minLoss)))
    if (config.save) {
      val modelPath = s"bin/${config.station}/${config.data}"
      uf.write.mode("overwrite").parquet(s"$modelPath/uf")
      vf.write.mode("overwrite").parquet(s"$modelPath/vf")
      bigdl.saveModel(s"$modelPath/${config.modelType}.bigdl", overWrite = true)
    }
    val startClock = System.currentTimeMillis()
    val model = estimator.fit(uf)
    val endClock = System.currentTimeMillis()
    // compute training errors and validation errors
    val predictionU = model.transform(uf)
    val predictionV = model.transform(vf)
    if (config.verbose) {
      predictionV.select("label", "prediction").show(false)
    }
    val errorU = computeError(predictionU)
    val maeU = errorU.head().getAs[DenseVector](0)
    val mseU = errorU.head().getAs[DenseVector](1)
    val errorV = computeError(predictionV)
    val maeV = errorV.head().getAs[DenseVector](0)
    val mseV = errorV.head().getAs[DenseVector](1)
    if (config.verbose) {
      println("avg(validationMAE) = " + maeV)
      println("avg(validationMSE) = " + mseV)
    }
    if (config.plot) {
      val spark = SparkSession.getActiveSession.get
      Plot.plot(spark, config, predictionV)
    }
    val trainingTime = (endClock - startClock) / 1000 // seconds
    val experimentConfig = config.modelType match {
      case 1 => ExperimentConfig(config.station, config.horizon, config.lookBack, config.nLayer,
        config.hiddenSize, config.epochs, config.dropoutRate, config.learningRate, modelType = config.modelType)
      case 2 => ExperimentConfig(config.station, config.horizon, config.lookBack, config.nLayer,
        config.hiddenSize, config.epochs, config.dropoutRate, config.learningRate, modelType = config.modelType,
        heads = config.nHead, blocks = config.nBlock, intermediateSize = config.intermediateSize)
    }
    Result(maeU.toArray, mseU.toArray, maeV.toArray, mseV.toArray, trainingTime, experimentConfig)
  }

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[Config](getClass.getName) {
      head(getClass.getName, "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("master, default is local[*]")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either {eval, train, predict}")
      opt[String]('d', "data").action((x, conf) => conf.copy(data = x)).text("data type: simple/complex")
      opt[String]('s', "station").action((x, conf) => conf.copy(station = x)).text("station viet-tri/vinh-yen/...")
      opt[Boolean]('p', "plot").action((_, conf) => conf.copy(plot = true)).text("plot figures")
      opt[Boolean]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode")
      opt[Boolean]('w', "write").action((_, conf) => conf.copy(save = true)).text("save data and trained model")
      opt[Int]('l', "lookBack").action((x, conf) => conf.copy(lookBack = x)).text("look-back (days)")
      opt[Int]('h', "horizon").action((x, conf) => conf.copy(horizon = x)).text("horizon (days)")
      opt[Boolean]('u', "bidirectional").action((_, conf) => conf.copy(bidirectional = true)).text("bidirectional LSTM")
      opt[Int]('j', "nLayer").action((x, conf) => conf.copy(nLayer = x)).text("number of layers")
      opt[Int]('r', "hiddenSize").action((x, conf) => conf.copy(hiddenSize = x)).text("hidden size (for both LSTM and BERT)")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Double]('a', "learningRate").action((x, conf) => conf.copy(learningRate = x)).text("learning rate")
      opt[Double]('o', "dropoutRate").action((x, conf) => conf.copy(dropoutRate = x)).text("dropout rate")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Int]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type: 1 = LSTM, 2 = LSTM+BERT")
      opt[Int]('x', "nHead").action((x, conf) => conf.copy(nHead = x)).text("number of attention heads of BERT")
      opt[Int]('y', "nBlock").action((x, conf) => conf.copy(nBlock = x)).text("number of blocks of BERT")
      opt[Int]('z', "bertSize").action((x, conf) => conf.copy(bertSize = x)).text("BERT output size")
      opt[Int]('i', "intermediateSize").action((x, conf) => conf.copy(intermediateSize = x)).text("FFN size of BERT")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        val conf = new SparkConf().setAppName(getClass.getName).setMaster(config.master)
          .set("spark.executor.memory", config.executorMemory).set("spark.driver.memory", config.driverMemory)
          .set("spark.driver.extraClassPath", "lib/mkl-java-x86_64-linux-2.0.0.jar")
        val sc = NNContext.initNNContext(conf)
        sc.setLogLevel("ERROR")
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
        // read data
        val ff = config.data match {
          case "clusterC" => DataReader.readClusterComplex(spark, s"dat/lnq/${config.station}-complex.csv")
          case "clusterS" => DataReader.readClusterSimple(spark, s"dat/lnq/${config.station}-simple.csv")
          case "complex" => DataReader.readComplex(spark, s"dat/lnq/${config.station}.csv")
          case "simple" => DataReader.readSimple(spark, "dat/lnq/y.80-19.tsv", config.station)
        }
        import upickle.default._
        implicit val configRW: ReadWriter[ExperimentConfig] = macroRW[ExperimentConfig]
        implicit val resultRW: ReadWriter[Result] = macroRW[Result]

        config.mode match {
          case "train" =>
            val result = train(ff, config)
            val json = write(result) + "\n"
            Files.write(Paths.get(s"dat/result-${config.data}-${config.station}.jsonl"), json.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
          case "eval" =>
            val modelPath = s"bin/${config.station}/" + (if (config.data == "complex") "c/" else "s/")
            val vf = spark.read.parquet(s"$modelPath/vf")
            vf.show(10)
            val bigdl = Models.loadModel[Float](s"$modelPath/${config.modelType}.bigdl")
            bigdl.summary()
            val prediction = bigdl.predict(vf, featureCols = Array("features"), predictionCol = "prediction")
            prediction.select("label", "prediction").show(false)
            if (config.plot) Plot.plot(spark, config, prediction)
            val error = computeError(prediction)
            val mae = error.head().getAs[DenseVector](0)
            val mse = error.head().getAs[DenseVector](1)
            println("avg(MAE) = " + mae)
            println("avg(MSE) = " + mse)
          case "roll" =>
            val ff = DataReader.readComplex(spark, s"dat/lnq/x.csv")
            val featureCols = Array("extra_0_0", "extra_0_1") ++ Array("month", "dayOfMonth")
            val af = roll(ff, config.lookBack, config.horizon, featureCols, "y", config.modelType == 2)
            ff.show()
            af.show()
          case "lstm" =>
            val horizons = Array(7)
            val lookBacks = Array(7)
            val layers = Array(2, 3, 4)
            val hiddenSizes = Array(128, 300, 400, 512)
            for {
              h <- horizons
              l <- lookBacks
              j <- layers
              r <- hiddenSizes
            } {
              for (_ <- 1 to 3) {
                val runConfig = Config(config.station, "train", data = config.data, lookBack =  l, horizon = h, nLayer = j,
                  hiddenSize = r, epochs = config.epochs, dropoutRate = config.dropoutRate, learningRate = config.learningRate,
                  batchSize = Runtime.getRuntime.availableProcessors * 4, driverMemory = config.driverMemory, executorMemory = config.executorMemory
                )
                val result = train(ff, runConfig)
                val json = write(result) + "\n"
                Files.write(Paths.get(s"dat/result-${config.data}-${config.station}.jsonl"), json.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
              }
            }
          case "bert" =>
            val horizons = Array(7)
            val lookBacks = Array(7)
            val layers = Array(5, 7)
            val hiddenSizes = Array(128, 300, 400, 512)
            val nBlocks = Array(2, 3, 4)
            val nHeads = Array(2, 4, 8)
            for {
              h <- horizons
              l <- lookBacks
              j <- layers
              r <- hiddenSizes
              x <- nHeads
              y <- nBlocks
            } {
              for (_ <- 1 to 3) {
                val runConfig = Config(config.station, "train", data = "complex", lookBack =  l, horizon = h, nLayer = j,
                  hiddenSize = r, epochs = config.epochs, dropoutRate = config.dropoutRate, learningRate = config.learningRate,
                  batchSize = Runtime.getRuntime.availableProcessors * 4, driverMemory = config.driverMemory, executorMemory = config.executorMemory,
                  nHead = x, nBlock = y, modelType = 2
                )
                val result = train(ff, runConfig)
                val json = write(result) + "\n"
                Files.write(Paths.get(s"dat/result-bert-${config.station}.jsonl"), json.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
              }
            }
        }
        spark.stop()
        println("Done!")
      case None => println("Invalid options!")
    }
  }
}

