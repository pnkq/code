package sol

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{RegexTokenizer, StringIndexer}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.layers.{Masking, Dense, Embedding, LSTM}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.{MSECriterion, SmoothL1Criterion}
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.{Trigger, Loss}
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.metrics.MSE
import org.apache.spark.ml.evaluation.RegressionEvaluator

/**
 * (C) phuonglh@gmail.com
 * 
 * An LSTM-based implementation of solubility prediction of protein. 
 *
 */
object SolubilityDL {
  private val numCores = Runtime.getRuntime.availableProcessors()

  val f = udf((seq: String) => seq.toCharArray().map(_.toString()))

  val hash = udf((tokens: Array[String], maxSeqLen: Int) => {
    val hs = tokens.map { token => AminoAcids.INDEX.getOrElse(token, 0) }
    if (hs.length < maxSeqLen) hs ++ Array.fill[Int](maxSeqLen - hs.length)(0) else hs.take(maxSeqLen)
  })

  private def createModel(maxSeqLen: Int, vocabSize: Int, embeddingSize: Int) = {
    val sequential = Sequential()
    sequential.add(Masking(0, inputShape = Shape(maxSeqLen)).setName("masking"))
    sequential.add(Embedding(inputDim = vocabSize + 1, outputDim = embeddingSize, inputLength = maxSeqLen).setName("embedding"))
    sequential.add(LSTM(16).setName("lstm"))
    sequential.add(Dense(1).setName("dense"))
  }

  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName(getClass.getName).setMaster("local[*]")
      .set("spark.executor.memory", "4g").set("spark.driver.memory", "8g")
    val sc = new SparkContext(conf)
    Engine.init

    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val basePath = "dat/sol/"
    val paths = Array("eSol_train", "eSol_test", "S.cerevisiae_test").map(p => s"$basePath/$p.csv")
    val train = spark.read.option("header", value = true).option("inferSchema", true).csv(paths(0))
    val valid = spark.read.option("header", value = true).option("inferSchema", true).csv(paths(1))
    train.show(false)

    val maxSeqLen = 1018
    val vocabSize = 20

    val ef = train.withColumn("tokens", f(col("sequence"))).withColumn("features", hash(col("tokens"), lit(maxSeqLen)))
    val efV = valid.withColumn("tokens", f(col("sequence"))).withColumn("features", hash(col("tokens"), lit(maxSeqLen)))
    ef.select("solubility", "sequence", "features").show(false)

    val (uf, vf) = (ef, efV)

    val model = createModel(maxSeqLen, vocabSize, 16)
    model.summary()

    val criterion = SmoothL1Criterion() // MSECriterion()
    val estimator = NNEstimator(model, criterion, Array(maxSeqLen), Array(1))
    val trainingSummary = TrainSummary(appName = "lstm", logDir = "sum/sol/")
    val validationSummary = ValidationSummary(appName = "lstm", logDir = "sum/sol/")
    val batchSize = numCores * 8
    estimator.setLabelCol("solubility").setFeaturesCol("features")
      .setBatchSize(batchSize)
      .setOptimMethod(new Adam(2E-4))
      .setMaxEpoch(5)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, vf, Array(new Loss(MSECriterion[Float]), new Loss(SmoothL1Criterion[Float]())), batchSize)
    estimator.fit(uf)

    val af = model.predict(uf, Array("features"), "prediction", batchSize).withColumn("z", explode(col("prediction")))
    val bf = model.predict(vf, Array("features"), "prediction", batchSize).withColumn("z", explode(col("prediction")))
    af.select("gene", "solubility", "prediction", "z").show(5, false)
    val evaluator = new RegressionEvaluator().setLabelCol("solubility").setPredictionCol("z").setMetricName("r2")
    var metric = evaluator.evaluate(af)
    println(s"R^2 on train data = $metric")
    metric = evaluator.evaluate(bf)
    println(s"R^2 on valid data = $metric")

    spark.stop()
  }
}
