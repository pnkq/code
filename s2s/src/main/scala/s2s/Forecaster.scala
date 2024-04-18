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
import org.apache.spark.ml.linalg.Vector
import com.cibo.evilplot.colors._
import com.cibo.evilplot.plot.renderers.PathRenderer

/**
 * An implementation of time series predictor using LSTM.
 */
object Forecaster {

  private def roll(ff: DataFrame, lookback: Int = 7, horizon: Int = 5): DataFrame = {
    // defining a window ordered by date
    val window = Window.orderBy("date")
    // create previous values, each in a separate column
    var gf = ff
    for (j <- -lookback to -1) {
      gf = gf.withColumn(s"y_$j", lag("y_0", -j).over(window))
    }
    gf.show()
    // create next values, each in a separate column
    for (j <- 1 to horizon) {
      gf = gf.withColumn(s"y_$j", lead("y_0", j).over(window))
    }
    // fill missing data (null) as zero
    val valueCols = ((-lookback to -1) ++ (1 to horizon)).map(c => s"y_$c")
    var af = gf
    for (c <- valueCols) {
      af = af.withColumn(c, when(isnull(col(c)), 0).otherwise(col(c)))
    }
    af
  }

  private def createModel(config: Config): Sequential[Float] = {
    val model = Sequential()
    model.add(Reshape(targetShape = Array(1, config.lookback + 3), inputShape = Shape(config.lookback + 3)).setName("reshape"))
    for (j <- 0 until config.numLayers - 1) {
      model.add(LSTM(config.hiddenSize, returnSequences = true).setName(s"lstm-$j"))
    }
    model.add(LSTM(config.hiddenSize).setName(s"lstm-${config.numLayers-1}"))
    model.add(Dropout(0.2).setName("dropout"))
    model.add(Dense(config.horizon).setName("dense"))
    model
  }

  private def plot(spark: SparkSession, config: Config, prediction: DataFrame) = {
    // draw multiple plots corresponding to number of time steps up to horizon for the first year of the validation set
    import spark.implicits._
    val ys = prediction.select("label").map(row => row.getAs[Vector](0).toDense.toArray).take(365).zipWithIndex
    val zs = prediction.select("prediction").map(row => row.getAs[Seq[Float]](0)).take(365).zipWithIndex
    val plots = (0 until config.horizon).map { d =>
      val dataY = ys.map(pair => Point(pair._2, pair._1(d)))
      val dataZ = zs.map(pair => Point(pair._2, pair._1(d)))
      Seq(
        LinePlot(dataY, Some(PathRenderer.named[Point](name = s"horizon=$d", strokeWidth = Some(1.2), color = HTMLNamedColors.gray)))
          .topPlot(LinePlot(dataZ, Some(PathRenderer.default[Point](strokeWidth = Some(1.2), color = Some(HTMLNamedColors.blue)))))
      )
    }
    // overlay plots
    val colors = Color.getGradientSeq(config.horizon)
    val days = (0 until config.horizon)
    val overlayPlots = days.map { d =>
      val dataZ = zs.map(pair => Point(pair._2, pair._1(d)))
      LinePlot(data = dataZ, Some(PathRenderer.named[Point](name = s"horizon=$d", strokeWidth = Some(1.2), color = colors(d))))
    }
    displayPlot(Facets(plots).standard().title("Rainfall in a Year").topLegend().render())
    displayPlot(Overlay(overlayPlots: _*).xAxis().xLabel("day").yAxis().yLabel("rainfall").yGrid().bottomLegend().render())
  }

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[Config](getClass().getName()) {
      head(getClass().getName(), "1.0")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 16g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either {eval, train, predict}")
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
        val conf = new SparkConf().setAppName(getClass.getName).setMaster("local[*]").set("spark.executor.memory", config.executorMemory).set("spark.driver.memory", config.driverMemory)
        val sc = NNContext.initNNContext(conf)
        sc.setLogLevel("ERROR")
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()

        config.mode match {
          case "train" =>
            val df = spark.read.options(Map("delimiter" -> "\t", "inferSchema" -> "true")).csv("dat/lnq/y.80-19.tsv")
            val stationCol = "_c8"
            val ef = df.select("_c0", "_c1", "_c2", stationCol)
            val prependZero = udf((x: Int) => if (x < 10) "0" + x.toString else x.toString)
            val ff = ef.withColumn("year", col("_c0").cast("string"))
              .withColumn("month", prependZero(col("_c1")))
              .withColumn("day", prependZero(col("_c2")))
              .withColumn("dateSt", concat_ws("/", col("year"), col("month"), col("day")))
              .withColumn("date", to_date(col("dateSt"), "yyy/MM/dd"))
              .withColumnRenamed(stationCol, "y_0")
            ff.show()
            ff.printSchema()

            val af = roll(ff, config.lookback, config.horizon)
            af.show()

            // assembler look-back vector and horizon vector
            val inputCols = (-config.lookback to 0).map(c => s"y_$c").toArray ++ Array("_c1", "_c2") // ++ Array(month, day)
            val assemblerX = new VectorAssembler().setInputCols(inputCols).setOutputCol("x")
            val assemblerY = new VectorAssembler().setInputCols((1 to config.horizon).map(c => s"y_$c").toArray).setOutputCol("y")
            val scalerX = new StandardScaler().setInputCol("x").setOutputCol("features")
            val scalerY = new StandardScaler().setInputCol("y").setOutputCol("label")
            val pipeline = new Pipeline().setStages(Array(assemblerX, assemblerY, scalerX, scalerY))
            val preprocesor = pipeline.fit(af)
            preprocesor.write.overwrite().save("bin/pre")

            val bf = preprocesor.transform(af)
            bf.show()

            // split roughly 80/20, keep order sequence
            val uf = bf.filter("_c0 <= 2012")
            val vf = bf.filter("2013 <= _c0")
            println(s"Training size = ${uf.count}")
            println(s"    Test size = ${vf.count}")
            uf.write.mode("overwrite").parquet("dat/uf")
            vf.write.mode("overwrite").parquet("dat/vf")

            val bigdl = createModel(config)
            bigdl.summary()
            val trainingSummary = TrainSummary(appName = config.modelType, logDir = "sum/")
            val validationSummary = ValidationSummary(appName = config.modelType, logDir = "sum/")

            val estimator = NNEstimator[Float](bigdl, new MSECriterion[Float](), featureSize = Array(config.lookback + 3), labelSize = Array(config.horizon))
            estimator.setBatchSize(config.batchSize).setOptimMethod(new Adam(lr = config.learningRate)).setMaxEpoch(config.epochs)
              .setTrainSummary(trainingSummary).setValidationSummary(validationSummary)
              .setValidation(Trigger.everyEpoch, vf, Array(new MAE[Float](), new Loss(new MSECriterion[Float]())), config.batchSize)
            val model = estimator.fit(uf)
            val prediction = model.transform(vf)
            prediction.select("label", "prediction").show(false)
            bigdl.saveModel(s"bin/${config.modelType}.bigdl")
            plot(spark, config, prediction)

          case "eval" =>
            val vf = spark.read.parquet("dat/vf")
            vf.show(10)
            val bigdl = Models.loadModel[Float](s"bin/${config.modelType}.bigdl")
            bigdl.summary()
            val prediction = bigdl.predict(vf, featureCols = Array("features"), predictionCol = "prediction")
            prediction.select("label", "prediction").show(false)
            plot(spark, config, prediction)
        }
        spark.stop()
      case None => println("Invalid options!")
    }
  }
}
