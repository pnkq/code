package vlp.s2s

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
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.MSECriterion
import com.intel.analytics.bigdl.dllib.optim.{Loss, Trigger}
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.StandardScaler

/**
 * An implementation of time series predictor using LSTM.
 */
object Forecaster {

  def roll(ff: DataFrame, lookback: Int = 7, horizon: Int = 5): DataFrame = {
    // defining a window ordered by date
    val window = Window.orderBy("date")
    // create previous values, each in a separate column
    var gf = ff
    for (j <- -lookback to -1) {
      gf = gf.withColumn(s"y($j)", lag("y(0)", -j).over(window))
    }
    gf.show()
    // create next values, each in a separate column
    for (j <- 1 to horizon) {
      gf = gf.withColumn(s"y($j)", lead("y(0)", j).over(window))
    }
    // fill missing data (null) as zero
    val valueCols = ((-lookback to -1) ++ (1 to horizon)).map(c => s"y($c)")
    var af = gf
    for (c <- valueCols) {
      af = af.withColumn(c, when(isnull(col(c)), 0).otherwise(col(c)))
    }
    af
  }

  def createModel(config: Config): Sequential[Float] = {
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

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName(getClass.getName).setMaster("local[*]")
    val sc = NNContext.initNNContext(conf)
    sc.setLogLevel("ERROR")
    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()

    val df = spark.read.options(Map("delimiter" -> "\t", "inferSchema" -> "true")).csv("dat/lnq/y.80-19.tsv")
    val stationCol = "_c8"
    val ef = df.select("_c0", "_c1", "_c2", stationCol)
    val prependZero = udf((x: Int) => if (x < 10) "0" + x.toString else x.toString)
    val ff = ef.withColumn("year", col("_c0").cast("string"))
      .withColumn("month", prependZero(col("_c1")))
      .withColumn("day", prependZero(col("_c2")))
      .withColumn("dateSt", concat_ws("/", col("year"), col("month"), col("day")))
      .withColumn("date", to_date(col("dateSt"), "yyy/MM/dd"))
      .withColumnRenamed(stationCol, "y(0)")
    ff.show()
    ff.printSchema()
    val config = Config()

    val af = roll(ff, config.lookback, config.horizon)
    af.show()
    // create a BigDL model
    // assembler look-back vector and horizon vector
    val inputCols = (-config.lookback to 0).map(c => s"y($c)").toArray ++ Array("_c1", "_c2") // ++ Array(month, day)
    val assemblerX = new VectorAssembler().setInputCols(inputCols).setOutputCol("x")
    val assemblerY = new VectorAssembler().setInputCols((1 to config.horizon).map(c => s"y($c)").toArray).setOutputCol("y")
    val scalerX = new StandardScaler().setInputCol("x").setOutputCol("features")
    val scalerY = new StandardScaler().setInputCol("y").setOutputCol("label")
    val pipeline = new Pipeline().setStages(Array(assemblerX, assemblerY, scalerX, scalerY))
    val preprocesor = pipeline.fit(af)
    val bf = preprocesor.transform(af)
    bf.show()

    // split roughly 80/20
    val uf = bf.filter("_c0 <= 2012")
    val vf = bf.filter("2013 <= _c0")
    println(s"Training size = ${uf.count}")
    println(s"    Test size = ${vf.count}")

    val bigdl = createModel(config)
    bigdl.summary()
    val trainingSummary = TrainSummary(appName = config.modelType, logDir = "sum/")
    val validationSummary = ValidationSummary(appName = config.modelType, logDir = "sum/")

    val estimator = NNEstimator[Float](bigdl, new MSECriterion[Float](), featureSize = Array(config.lookback + 3), labelSize = Array(config.horizon))
    estimator.setBatchSize(config.batchSize)
      .setOptimMethod(new Adam(lr = config.learningRate))
      .setMaxEpoch(config.epochs)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, vf, Array(new MAE[Float](), new Loss(new MSECriterion[Float]())), config.batchSize)
    val model = estimator.fit(uf)
    val cf = model.transform(vf)
    cf.select("label", "prediction").show(false)
    spark.stop()
  }
}
