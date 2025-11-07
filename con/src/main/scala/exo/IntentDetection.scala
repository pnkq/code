package exo

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
import com.intel.analytics.bigdl.dllib.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.{Top1Accuracy, Trigger}
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.dllib.utils.Shape

/**
 * (C) phuonglh@gmail.com
 * 
 * An LSTM-based implementation of intent classifier on the HWU dataset.
 *
 */
object IntentDetection {
  private val numCores = Runtime.getRuntime.availableProcessors()

  val hash = udf((tokens: Array[String], vocabSize: Int, maxSeqLen: Int) => {
    val hs = tokens.map { token =>
      val utf8 = UTF8String.fromString(token)
      val rawIdx = hashUnsafeBytes2(utf8.getBaseObject, utf8.getBaseOffset, utf8.numBytes(), 43)
      val rawMod = rawIdx % vocabSize
      rawMod + (if (rawMod < 0) vocabSize else 0)
    }
    if (hs.length < maxSeqLen) hs ++ Array.fill[Int](maxSeqLen - hs.length)(0) else hs.take(maxSeqLen)
  })

  val inc = udf((v: Double) => v + 1)

  private def createPreprocessor(df: DataFrame) = {
    val indexer = new StringIndexer().setInputCol("category").setOutputCol("index")
    val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern("""[\s,.:!']+""")
    val pipeline = new Pipeline().setStages(Array(indexer, tokenizer))
    pipeline.fit(df)
  }

  private def createModel(maxSeqLen: Int, vocabSize: Int, embeddingSize: Int, labelSize: Int) = {
    val sequential = Sequential()
    sequential.add(Masking(0, inputShape = Shape(maxSeqLen)))
    sequential.add(Embedding(inputDim = vocabSize, outputDim = embeddingSize, inputLength = maxSeqLen))
    sequential.add(LSTM(64))
    sequential.add(Dense(labelSize, activation = "softmax"))
  }

  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName(getClass.getName).setMaster("local[4]")
      .set("spark.executor.memory", "4g").set("spark.driver.memory", "8g")
    val sc = new SparkContext(conf)
    Engine.init

    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val basePath = "dat/hwu/"
    val paths = Array("train", "val", "test").map(p => s"$basePath/$p.csv")
    val train = spark.read.option("header", value = true).csv(paths(0))
    val valid = spark.read.option("header", value = true).csv(paths(1))
    val preprocessor = createPreprocessor(train)
    val df = preprocessor.transform(train)
    val dfV = preprocessor.transform(valid)
    df.show(false)

    val maxSeqLen = 25
    val vocabSize = 8192

    val ef = df.withColumn("features", hash(col("tokens"), lit(vocabSize), lit(maxSeqLen))).withColumn("label", inc(col("index")))
    ef.select("label", "features").show(false)
    val efV = dfV.withColumn("features", hash(col("tokens"), lit(vocabSize), lit(maxSeqLen))).withColumn("label", inc(col("index")))

    val (uf, vf) = (ef, efV)

    val model = createModel(maxSeqLen, vocabSize, 100, 64)
    model.summary()

    val criterion = ClassNLLCriterion(sizeAverage = false, logProbAsInput = false)
    val estimator = NNEstimator(model, criterion, Array(maxSeqLen), Array(1))
    val trainingSummary = TrainSummary(appName = "lstm2", logDir = "sum/hwu/")
    val validationSummary = ValidationSummary(appName = "lstm2", logDir = "sum/hwu/")
    val batchSize = numCores * 8
    estimator.setLabelCol("label").setFeaturesCol("features")
      .setBatchSize(batchSize)
      .setOptimMethod(new Adam(2E-4))
      .setMaxEpoch(40)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, vf, Array(new Top1Accuracy[Float]()), batchSize)
    estimator.fit(uf)
    spark.stop()
  }
}
