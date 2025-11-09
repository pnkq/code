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
import com.intel.analytics.bigdl.dllib.nn.TimeDistributedMaskCriterion
import vlp.dep.TimeDistributedTop1Accuracy
import com.intel.analytics.bigdl.dllib.keras.layers.Bidirectional

/**
 * (C) phuonglh@gmail.com
 * 
 * An LSTM-based implementation of a part-of-speech tagger on a UD English treebank.
 *
 */
object Tagger {
  private val numCores = Runtime.getRuntime.availableProcessors()

  val tags = Seq("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", 
    "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X").zipWithIndex.map(p => (p._1, p._2 + 1)).toMap

  val hash = udf((tokens: Array[String], vocabSize: Int, maxSeqLen: Int) => {
    val hs = tokens.map { token =>
      val utf8 = UTF8String.fromString(token)
      val rawIdx = hashUnsafeBytes2(utf8.getBaseObject, utf8.getBaseOffset, utf8.numBytes(), 43)
      val rawMod = rawIdx % vocabSize
      rawMod + (if (rawMod < 0) vocabSize else 0)
    }
    if (hs.length < maxSeqLen) hs ++ Array.fill[Int](maxSeqLen - hs.length)(0) else hs.take(maxSeqLen)
  })

  val index = udf((labels: Array[String], maxSeqLen: Int) => {
    val ys = labels.map(token => tags(token))
    if (ys.length < maxSeqLen) ys ++ Array.fill[Int](maxSeqLen - ys.length)(-1) else ys.take(maxSeqLen)
  })

  private def createModel(maxSeqLen: Int, vocabSize: Int, embeddingSize: Int, labelSize: Int) = {
    val sequential = Sequential()
    sequential.add(Masking(0, inputShape = Shape(maxSeqLen)).setName("masking"))
    sequential.add(Embedding(inputDim = vocabSize, outputDim = embeddingSize, inputLength = maxSeqLen).setName("embedding"))
    sequential.add(Bidirectional(LSTM(64, returnSequences = true).setName("lstm")).setName("biLSTM"))
    sequential.add(Dense(labelSize, activation = "softmax").setName("softmax"))
  }

  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName(getClass.getName).setMaster("local[*]")
      .set("spark.executor.memory", "4g").set("spark.driver.memory", "8g")
    val sc = new SparkContext(conf)
    Engine.init

    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val basePath = "dat/dep/UD_English-EWT/en_ewt-ud-"
    val paths = Array("train", "dev", "test").map(p => s"$basePath$p.jsonl")
    val train = spark.read.json(paths(0))
    val valid = spark.read.json(paths(1))
    valid.select("words", "tags").show(5, false)

    val vocabSize = 8192*2
    val maxSeqLen = 30

    val uf = train.withColumn("features", hash(col("words"), lit(vocabSize), lit(maxSeqLen))).withColumn("label", index(col("tags"), lit(30)))
    val vf = valid.withColumn("features", hash(col("words"), lit(vocabSize), lit(maxSeqLen))).withColumn("label", index(col("tags"), lit(30)))
    vf.select("features", "label").show(5, false)

    val model = createModel(maxSeqLen, vocabSize, 100, tags.size)
    model.summary()

    val criterion = TimeDistributedMaskCriterion(
      ClassNLLCriterion(sizeAverage = false, logProbAsInput = false, paddingValue = -1), 
      paddingValue = -1
    )
    val estimator = NNEstimator(model, criterion, Array(maxSeqLen), Array(maxSeqLen))
    val trainingSummary = TrainSummary(appName = "lstm2", logDir = "sum/tag/")
    val validationSummary = ValidationSummary(appName = "lstm2", logDir = "sum/tag/")
    val batchSize = numCores * 8
    estimator.setLabelCol("label").setFeaturesCol("features")
      .setBatchSize(batchSize)
      .setOptimMethod(new Adam(2E-4))
      .setMaxEpoch(20)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, vf, Array(new TimeDistributedTop1Accuracy(-1)), batchSize)
    estimator.fit(uf)
    spark.stop()
  }
}
