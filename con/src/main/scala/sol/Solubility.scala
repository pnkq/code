package sol

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{RegexTokenizer, StringIndexer}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.unsafe.hash.Murmur3_x86_32.hashUnsafeBytes2
import org.apache.spark.unsafe.types.UTF8String
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.layers.{Masking, Dense, Embedding, LSTM}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.MSECriterion
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.{Trigger, Loss}
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.metrics.MSE

/**
 * (C) phuonglh@gmail.com
 * 
 * An LSTM-based implementation of solubility prediction of protein. 
 *
 */
object Solubility {
  private val numCores = Runtime.getRuntime.availableProcessors()

  val f = udf((seq: String) => seq.toCharArray().map(_.toString()))

  val hash = udf((tokens: Array[String], vocabSize: Int, maxSeqLen: Int) => {
    val hs = tokens.map { token =>
      val utf8 = UTF8String.fromString(token)
      val rawIdx = hashUnsafeBytes2(utf8.getBaseObject, utf8.getBaseOffset, utf8.numBytes(), 43)
      val rawMod = rawIdx % vocabSize
      rawMod + (if (rawMod < 0) vocabSize else 0)
    }
    if (hs.length < maxSeqLen) hs ++ Array.fill[Int](maxSeqLen - hs.length)(0) else hs.take(maxSeqLen)
  })

  private def createModel(maxSeqLen: Int, vocabSize: Int, embeddingSize: Int) = {
    val sequential = Sequential()
    sequential.add(Masking(0, inputShape = Shape(maxSeqLen)))
    sequential.add(Embedding(inputDim = vocabSize, outputDim = embeddingSize, inputLength = maxSeqLen))
    sequential.add(LSTM(64))
    sequential.add(Dense(1, activation = "softmax"))
  }

  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName(getClass.getName).setMaster("local[4]")
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

    val ef = train.withColumn("tokens", f(col("sequence"))).withColumn("features", hash(col("tokens"), lit(vocabSize), lit(maxSeqLen)))
    val efV = valid.withColumn("tokens", f(col("sequence"))).withColumn("features", hash(col("tokens"), lit(vocabSize), lit(maxSeqLen)))
    ef.select("solubility", "features").show(false)

    val (uf, vf) = (ef, efV)

    val model = createModel(maxSeqLen, vocabSize, 32)
    model.summary()

    val criterion = MSECriterion()
    val estimator = NNEstimator(model, criterion, Array(maxSeqLen), Array(1))
    val trainingSummary = TrainSummary(appName = "lstm", logDir = "sum/sol/")
    val validationSummary = ValidationSummary(appName = "lstm", logDir = "sum/sol/")
    val batchSize = numCores * 8
    estimator.setLabelCol("solubility").setFeaturesCol("features")
      .setBatchSize(batchSize)
      .setOptimMethod(new Adam(2E-4))
      .setMaxEpoch(20)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, vf, Array(new Loss(MSECriterion[Float])), batchSize)
    // estimator.fit(uf)

    spark.stop()
  }
}
