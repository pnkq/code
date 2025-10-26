package vlp.dep

import org.apache.log4j.{Logger, Level}
import org.slf4j.LoggerFactory
import org.apache.spark.sql.SparkSession
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.models.{Model, Models, KerasNet}
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._
import com.intel.analytics.bigdl.dllib.nn.TimeDistributedMaskCriterion
import com.intel.analytics.bigdl.dllib.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.optim.Trigger
import vlp.ner.TimeDistributedTop1Accuracy

case class PretrainerConfig(
  language: String = "eng",
  maxSeqLen: Int = 80,
  embeddingSize: Int = 20,
  layers: Int = 2,
  heads: Int = 4,
  hiddenSize: Int = 100,
)

/**
 * (C) phuonglh@gmail.com, October 26, 2025
 *
 *  
 */
object TransitionPretrainer {
  val logger = LoggerFactory.getLogger(getClass.getName)

  def createModel(config: PretrainerConfig, vocabSize: Int): KerasNet[Float] = {
    // Create a BERT encoder using one input tensor of size 4*maxSeqLen 
    // and output maxSeqLen of softmax distributions
    val input = Input(inputShape = Shape(4*config.maxSeqLen), name = "input")
    val reshape = Reshape(targetShape = Array(4, config.maxSeqLen)).setName("reshape").inputs(input)
    val split = SplitTensor(1, 4).setName("split").inputs(reshape)
    val selectIds = SelectTable(0).setName("inputId").inputs(split)
    val inputIds = Squeeze(1).setName("squeezeId").inputs(selectIds)
    val selectSegments = SelectTable(1).setName("segmentId").inputs(split)
    val segmentIds = Squeeze(1).setName("squeezeSegment").inputs(selectSegments)
    val selectPositions = SelectTable(2).setName("positionId").inputs(split)
    val positionIds = Squeeze(1).setName("squeezePosition").inputs(selectPositions)
    val selectMasks = SelectTable(3).setName("masks").inputs(split)
    val masksReshaped = Reshape(targetShape = Array(1, 1, config.maxSeqLen)).setName("mask").inputs(selectMasks)
    // reserve 0f for (right) padding, hence vocabSize + 1
    val bert = BERT(vocab = vocabSize + 1, hiddenSize = config.embeddingSize, nBlock = config.layers, nHead = config.heads, maxPositionLen = config.maxSeqLen,
      intermediateSize = config.hiddenSize, outputAllBlock = false).setName("bert")
    val bertNode = bert.inputs(Array(inputIds, segmentIds, positionIds, masksReshaped))
    val outputBERT = SelectTable(0).setName("firstBlock").inputs(bertNode)
    val output = Dense(vocabSize, activation="softmax").setName("softmax").inputs(outputBERT)
    Model(input, output)
  }

  def train(model: KerasNet[Float], config: PretrainerConfig, train: DataFrame, valid: DataFrame) = {
    val (featureSize, labelSize) = (Array(Array(4*config.maxSeqLen)), Array(config.maxSeqLen))
    val estimator = NNEstimator(model, TimeDistributedMaskCriterion(ClassNLLCriterion(sizeAverage = false, paddingValue = -1), paddingValue = -1), featureSize, labelSize)
    val trainingSummary = TrainSummary(appName = "trp", logDir = s"sum/dep/${config.language}")
    val validationSummary = ValidationSummary(appName = "trp", logDir = s"sum/dep/${config.language}")
    val batchSize = Runtime.getRuntime.availableProcessors() * 16
    estimator.setLabelCol("y").setFeaturesCol("features")
      .setBatchSize(batchSize)
      .setOptimMethod(new Adam(1E-4))
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, valid, Array(new TimeDistributedTop1Accuracy(-1)), batchSize)
      .setEndWhen(Trigger.or(Trigger.maxEpoch(40), Trigger.minLoss(0.1f)))
    estimator.fit(train)
    model.saveModel(s"bin/dep/trp/${config.language}.bigdl", overWrite = true)
  }

  def createPreprocessor(df: DataFrame, config: PretrainerConfig) = {
    val vectorizer = new CountVectorizer().setInputCol("transitions").setOutputCol("vector")
    val vectorizerModel = vectorizer.fit(df)
    // index the transitions, reserve 0 for padding value
    val vocabMap = vectorizerModel.vocabulary.zipWithIndex.map{ case (w, i) => (w, i+1) }.toMap
    logger.info(vocabMap.toString)

    // create a udf to transform a sparse (count) vector into a sequence of transition indices
    // right pad with 0 values. Add segment ids, position ids and mask ids for BERT
    val f = udf((xs: Seq[String]) => {
      val seq = xs.map(x => vocabMap.getOrElse(x, 0))
      val ids = if (seq.length >= config.maxSeqLen) 
        seq.take(config.maxSeqLen)
      else
        seq ++ Seq.fill[Int](config.maxSeqLen - seq.length)(0)
      val segments = Seq.fill[Int](config.maxSeqLen)(0)
      val positions = if (seq.length >= config.maxSeqLen)
        (0 until config.maxSeqLen).toSeq
      else
        (0 until seq.length).toSeq ++ Seq.fill[Int](config.maxSeqLen - seq.length)(0)
      val masks = if (seq.length >= config.maxSeqLen)
        Seq.fill[Int](config.maxSeqLen)(1)
      else
        Seq.fill[Int](seq.length)(1) ++ Seq.fill[Int](config.maxSeqLen - seq.length)(0)
      ids ++ segments ++ positions ++ masks
    })
    // transform the input data frame
    val ef = df.withColumn("features", f(col("transitions")))
    ef.show(10)
    logger.info(ef.select("features").head.toString)

    vocabMap.size
  }


  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName(getClass.getName)
      .master("local[*]")
      .config("spark.driver.host", "localhost")
      .config("spark.driver.memory", "8g")
      .config("spark.shuffle.blockTransferService", "nio")
      .getOrCreate()
    Engine.init
    val config = PretrainerConfig()
    val df = spark.read.json("dat/dep/en-as-dev.jsonl")
    df.show(10)
    val vocabSize = createPreprocessor(df, config)
    val model = createModel(config, vocabSize)
    model.summary()

    spark.stop()
  }
}
