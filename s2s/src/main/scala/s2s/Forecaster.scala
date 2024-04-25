package s2s

import com.intel.analytics.bigdl.dllib.NNContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.keras.metrics.MAE
import com.intel.analytics.bigdl.dllib.keras.models.Models
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.MSECriterion
import com.intel.analytics.bigdl.dllib.optim.{Loss, Trigger}
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.StandardScaler
import scopt.OptionParser
import com.cibo.evilplot.displayPlot
import com.cibo.evilplot.plot._
import com.cibo.evilplot.plot.aesthetics.DefaultTheme._
import com.cibo.evilplot.numeric.Point
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import com.cibo.evilplot.colors._
import com.cibo.evilplot.plot.renderers.PathRenderer

/**
 * An implementation of time series predictor using LSTM.
 */
object Forecaster {

  /**
   * Rolls a data frame based on the "date" column.
   * @param ff a data frame
   * @param lookback number of past days to look back
   * @param horizon number of future days to forecast
   * @param featureCols column names to row
   * @param targetCol target column
   * @return a data frame with an array of columns for input features
   */
  private def roll(ff: DataFrame, lookback: Int, horizon: Int, featureCols: Array[String], targetCol: String = "y", imputeMissing: Boolean = false): DataFrame = {
    // defining a window ordered by date
    val window = Window.orderBy("date")
    // roll values, each value is in a separate column
    var gf = ff
    // all features and the target value are rolled backward (NOTE: append targetCol)
    for (name <- featureCols :+ targetCol) {
      for (j <- -lookback + 1 to -1) {
        gf = gf.withColumn(s"${name}_$j", lag(name, -j).over(window))
      }
    }
    // only target value is rolled forward
    for (j <- 1 to horizon) {
      gf = gf.withColumn(s"${targetCol}_$j", lead(targetCol, j).over(window))
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

  private def createModel(config: Config, featureSize: Int): Sequential[Float] = {
    val model = Sequential()
    model.add(Reshape(targetShape = Array(config.lookback, featureSize), inputShape = Shape(config.lookback * featureSize)).setName("reshape"))
    if (!config.bidirectional) {
      for (j <- 0 until config.numLayers - 1) {
        model.add(LSTM(config.hiddenSize, returnSequences = true).setName(s"lstm-$j"))
      }
      model.add(LSTM(config.hiddenSize).setName(s"lstm-${config.numLayers-1}"))
    } else {
      for (j <- 0 until config.numLayers - 1) {
        model.add(Bidirectional(LSTM(config.hiddenSize, returnSequences = true).setName(s"lstm-$j")))
      }
      model.add(Bidirectional(LSTM(config.hiddenSize).setName(s"lstm-${config.numLayers-1}")))
    }
    model.add(Dropout(config.dropoutRate).setName("dropout"))
    model.add(Dense(config.horizon).setName("dense"))
    model
  }

  private def plot(spark: SparkSession, config: Config, prediction: DataFrame): Unit = {
    // draw multiple plots corresponding to number of time steps up to horizon for the first year of the validation set
    import spark.implicits._
    val ys = prediction.select("label").map(row => row.getAs[Vector](0).toDense.toArray).take(365).zipWithIndex
    val zs = prediction.select("prediction").map(row => row.getAs[Seq[Float]](0)).take(365).zipWithIndex

    // plot +1 day prediction and label
    val dataY0 = ys.map(pair => Point(pair._2, pair._1.head))
    val dataZ0 = zs.map(pair => Point(pair._2, pair._1.head))
    displayPlot(Overlay(
      LinePlot(dataY0, Some(PathRenderer.named[Point](name = "horizon=0", strokeWidth = Some(1.2), color = HTMLNamedColors.gray))),
      LinePlot(dataZ0, Some(PathRenderer.default[Point](strokeWidth = Some(1.2), color = Some(HTMLNamedColors.blue))))
    ).xAxis().xLabel("day").yAxis().yLabel("rainfall").yGrid().bottomLegend())

    val plots = (0 until config.horizon).map { d =>
      val dataY = ys.map(pair => Point(pair._2, pair._1(d)))
      val dataZ = zs.map(pair => Point(pair._2, pair._1(d)))
      Seq(
        LinePlot(dataY, Some(PathRenderer.named[Point](name = s"horizon=$d", strokeWidth = Some(1.2), color = HTMLNamedColors.gray)))
          .topPlot(LinePlot(dataZ, Some(PathRenderer.default[Point](strokeWidth = Some(1.2), color = Some(HTMLNamedColors.blue)))))
      )
    }
    displayPlot(Facets(plots).standard().title("Rainfall in a Year").topLegend().render())

    // overlay plots
    val colors = Color.getGradientSeq(config.horizon)
    val days = 0 until config.horizon
    val overlayPlots = days.map { d =>
      val dataZ = zs.map(pair => Point(pair._2, pair._1(d)))
      LinePlot(data = dataZ, Some(PathRenderer.named[Point](name = s"horizon=$d", strokeWidth = Some(1.2), color = colors(d))))
    }
    displayPlot(Overlay(overlayPlots: _*).xAxis().xLabel("day").yAxis().yLabel("rainfall").yGrid().bottomLegend())
  }

  private def readSimple(spark: SparkSession, path: String) = {
    val df = spark.read.options(Map("delimiter" -> "\t", "inferSchema" -> "true")).csv(path)
    val stationCol = "_c8"
    val ef = df.select("_c0", "_c1", "_c2", stationCol)
    val prependZero = udf((x: Int) => if (x < 10) "0" + x.toString else x.toString)
    val ff = ef.withColumn("year", col("_c0"))
      .withColumn("yearSt", col("_c0").cast("string"))
      .withColumn("monthSt", prependZero(col("_c1")))
      .withColumn("daySt", prependZero(col("_c2")))
      .withColumn("dateSt", concat_ws("/", col("yearSt"), col("monthSt"), col("daySt")))
      .withColumn("date", to_date(col("dateSt"), "yyy/MM/dd"))
      .withColumnRenamed(stationCol, "y")
    (ff, Array("_c1", "_c2")) // Array(month, dayOfMonth)
  }
  /**
   * Reads a complex CSV file containing more than a hundred of columns. The label (rainfall) column is named "y".
   * Should filter out data of year < 2020 (many missing re-analysis data).
   * @param spark spark session
   * @param path a path to the CSV file
   * @return a data frame and an array of date columns
   */
  private def readComplex(spark: SparkSession, path: String) = {
    val cf = spark.read.options(Map("inferSchema" -> "true", "header" -> "true")).csv(path)
    val selectedColNames = cf.schema.fieldNames.filter(name =>
      name.contains("_0_") || name.contains("_1_") || name.contains("_2_")
    )
    val df = cf.select((Array("Date", "y") ++ selectedColNames).map(name => col(name)) :_*)
    val ef = df.withColumn("date", to_date(col("Date"), "yyy-MM-dd"))
      .withColumn("year", year(col("date")))
      .withColumn("month", month(col("date")))
      .withColumn("dayOfMonth", dayofmonth(col("date")))
    val ff = ef.filter("year < 2020")
    (ff, Array("month", "dayOfMonth"))
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
    output.show(false)
    import org.apache.spark.ml.stat.Summarizer
    output.select(Summarizer.mean(col("mae")), Summarizer.mean(col("mse")))
  }

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[Config](getClass.getName) {
      head(getClass.getName, "1.0")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 16g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either {eval, train, predict}")
      opt[String]('d', "data").action((x, conf) => conf.copy(data = x)).text("data type: simple/complex")
      opt[String]('s', "station").action((x, conf) => conf.copy(station = x)).text("station viet-tri/vinh-yen/...")
      opt[Boolean]('u', "u").action((_, conf) => conf.copy(bidirectional = true)).text("bidirectional RNN")
      opt[Boolean]('p', "p").action((_, conf) => conf.copy(plot = true)).text("plot figures")
      opt[Int]('l', "lookBack").action((x, conf) => conf.copy(lookback = x)).text("look-back (days)")
      opt[Int]('h', "horizon").action((x, conf) => conf.copy(horizon = x)).text("horizon (days)")
      opt[Int]('j', "numLayers").action((x, conf) => conf.copy(numLayers = x)).text("number of layers")
      opt[Int]('r', "hiddenSize").action((x, conf) => conf.copy(hiddenSize = x)).text("hidden size")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Double]('a', "learningRate").action((x, conf) => conf.copy(learningRate = x)).text("learning rate")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        val conf = new SparkConf().setAppName(getClass.getName).setMaster("local[*]")
          .set("spark.executor.memory", config.executorMemory).set("spark.driver.memory", config.driverMemory)
        val sc = NNContext.initNNContext(conf)
        sc.setLogLevel("ERROR")
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()

        val (ff, dateInputCols) = config.data match {
          case "complex" => readComplex(spark, s"dat/lnq/${config.station}.csv")
          case "simple" => readSimple(spark, "dat/lnq/y.80-19.tsv")
        }
        val modelPath = s"bin/${config.station}/" + (if (config.data == "complex") "c/" else "s/")
        val targetCol = "y"
        val extraCols = ff.schema.fieldNames.filter(name => name.contains("extra"))
        val featureCols = extraCols ++ dateInputCols

        config.mode match {
          case "train" =>
            ff.show()
            val af = roll(ff, config.lookback, config.horizon, featureCols, targetCol)
            if (config.data == "simple") af.show()
            // arrange input by time steps (i.e., -3, -2, -1, 0)
            val inputCols = (-config.lookback + 1 until 0).toArray.flatMap { j =>
              af.schema.fieldNames.filter(name => name.endsWith(j.toString))
            } ++ featureCols ++ Array(targetCol)
            val assemblerX = new VectorAssembler().setInputCols(inputCols).setOutputCol("input").setHandleInvalid("skip")
            val assemblerY = new VectorAssembler().setInputCols((1 to config.horizon).map(c => s"y_$c").toArray).setOutputCol("output").setHandleInvalid("skip")
            val scalerX = new StandardScaler().setInputCol("input").setOutputCol("features")
            val scalerY = new StandardScaler().setInputCol("output").setOutputCol("label")
            val pipeline = new Pipeline().setStages(Array(assemblerX, assemblerY, scalerX, scalerY))
            val preprocessor = pipeline.fit(af)
            preprocessor.write.overwrite().save(s"$modelPath/pre")

            val bf = preprocessor.transform(af)
            bf.select("input", "output").show()

            // split roughly 90/10, keep sequence order
            val uf = bf.filter("year <= 2015")
            val vf = bf.filter("2016 <= year")
            println(s"Training size = ${uf.count}")
            println(s"    Test size = ${vf.count}")
            uf.write.mode("overwrite").parquet(s"$modelPath/uf")
            vf.write.mode("overwrite").parquet(s"$modelPath/vf")

            val featureSize = featureCols.length + 1
            val bigdl = createModel(config, featureSize)
            bigdl.summary()
            val trainingSummary = TrainSummary(appName = config.modelType, logDir = s"sum/${config.station}")
            val validationSummary = ValidationSummary(appName = config.modelType, logDir = s"sum/${config.station}")

            val estimator = NNEstimator[Float](bigdl, new MSECriterion[Float](), featureSize = Array(featureSize * config.lookback), labelSize = Array(config.horizon))
            estimator.setBatchSize(config.batchSize).setOptimMethod(new Adam(lr = config.learningRate)).setMaxEpoch(config.epochs)
              .setTrainSummary(trainingSummary).setValidationSummary(validationSummary)
              .setValidation(Trigger.everyEpoch, vf, Array(new MAE[Float](), new Loss(new MSECriterion[Float]())), config.batchSize)
            val model = estimator.fit(uf)
            val prediction = model.transform(vf)
            prediction.select("label", "prediction").show(false)
            bigdl.saveModel(s"$modelPath/${config.modelType}.bigdl", overWrite = true)
            if (config.plot) plot(spark, config, prediction)
            val error = computeError(prediction)
            val mae = error.head().getAs[DenseVector](0)
            val mse = error.head().getAs[DenseVector](1)
            println("avg(MAE) = " + mae)
            println("avg(MSE) = " + mse)
          case "eval" =>
            val vf = spark.read.parquet(s"$modelPath/vf")
            vf.show(10)
            val bigdl = Models.loadModel[Float](s"$modelPath/${config.modelType}.bigdl")
            bigdl.summary()
            val prediction = bigdl.predict(vf, featureCols = Array("features"), predictionCol = "prediction")
            prediction.select("label", "prediction").show(false)
            if (config.plot) plot(spark, config, prediction)
            val error = computeError(prediction)
            val mae = error.head().getAs[DenseVector](0)
            val mse = error.head().getAs[DenseVector](1)
            println("avg(MAE) = " + mae)
            println("avg(MSE) = " + mse)
          case "roll" =>
            val (ff, _) = readComplex(spark, s"dat/lnq/x.csv")
            val featureCols = Array("extra_0_0", "extra_0_1") ++ Array("month", "dayOfMonth")
            val af = roll(ff, config.lookback, config.horizon, featureCols)
            ff.show()
            af.show()
        }
        spark.stop()
        println("Done!")
      case None => println("Invalid options!")
    }
  }
}

