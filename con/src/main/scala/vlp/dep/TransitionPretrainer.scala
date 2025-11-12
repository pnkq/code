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
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.optim.{AccuracyResult, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

case class PretrainerConfig(
  language: String = "eng",
  system: String = "as", // ae
  maxSeqLen: Int = 50,
  embeddingSize: Int = 20,
  layers: Int = 2,
  heads: Int = 4,
  hiddenSize: Int = 100,
  maskedRatio: Double = 0.3,
  maxIters: Int = 10
)

class TimeDistributedTop1Accuracy(paddingValue: Int = -1)(implicit ev: TensorNumeric[Float]) extends ValidationMethod[Float] {
  override def apply(output: Activity, target: Activity): ValidationResult = {
    var correct = 0
    var count = 0
    val _output = output.asInstanceOf[Tensor[Float]] // nDim = 3
    val _target = target.asInstanceOf[Tensor[Float]] // nDim = 2
    // split by batch size (dim = 1 of output and target)
    _output.split(1).zip(_target.split(1))
      .foreach { case (tensor, ys) =>
      // split by time slice (dim = 1 of tensor)
      val zs = tensor.split(1).map { t =>
        val (_, k) = t.max(1) // the label with max score
        k(Array(1)).toInt // k is a tensor => extract its value
      }
      // filter the padded value (-1) in the target before matching with the output
      val c = ys.toArray().map(_.toInt).zip(zs).filter(p => p._1 != paddingValue)
        .map(p => if (p._1 == p._2) 1 else 0)
      correct += c.sum
      count += c.size
    }
    new AccuracyResult(correct, count)
  }
  override def format(): String = "Time Distributed Top1Accuracy"
}

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
    val trainingSummary = TrainSummary(appName = s"${config.language}", logDir = s"sum/asp/")
    val validationSummary = ValidationSummary(appName = s"${config.language}", logDir = s"sum/asp/")
    val batchSize = Runtime.getRuntime.availableProcessors() * 16
    estimator.setLabelCol("label").setFeaturesCol("features")
      .setBatchSize(batchSize)
      .setOptimMethod(new Adam(1E-4))
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, valid, Array(new TimeDistributedTop1Accuracy(-1)), batchSize)
      .setEndWhen(Trigger.maxEpoch(config.maxIters))
    estimator.fit(train)
    model.saveModel(s"bin/asp/${config.language}.bigdl", overWrite = true)
  }

  def preprocess(df: DataFrame, dfV: DataFrame, config: PretrainerConfig) = {
    val vectorizer = new CountVectorizer().setInputCol("transitions")
    val vectorizerModel = vectorizer.fit(df)
    // index the transitions, reserve 0 for padding value
    val vocabMap = vectorizerModel.vocabulary.zipWithIndex.map{ case (w, i) => (w, i+1) }.toMap
    logger.info(vocabMap.toString)

    // create a udf to transform a sparse (count) vector into a sequence of transition indices
    // right pad with 0 values. Add segment ids, position ids and mask ids for BERT
    // in the Arc-Standard system, if the sentence has more than 2 tokens, the first transition is always SH,
    // therefore, we don't include the first transition into our features.
    // Also, the last transition is also always a SH (to make the parsing config. final.)
    val f = udf((xs: Seq[String]) => {
      val seq = xs.slice(1, xs.length-1).map(x => vocabMap.getOrElse(x, 0))
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
    // transform the input data frame to have the "features" column
    val ef = df.withColumn("features", f(col("transitions")))
    val efV = dfV.withColumn("features", f(col("transitions")))
    ef.show(10)
    logger.info(ef.select("features").head.toString)

    // prepare the label sequence using a padding value of -1
    // we use the masked language modeling (MLM) objective
    val random = new scala.util.Random(220712)
    val g = udf((xs: Seq[String]) => {
      val seq = xs.slice(1, xs.length-1).map(x => vocabMap.getOrElse(x, 0)) // 0 label will results in an error in BigDL
      val ys = seq.map { s => 
        val keepProb = random.nextDouble()
        if (keepProb <= config.maskedRatio) s else -1
      }
      if (ys.length >= config.maxSeqLen)
        ys.take(config.maxSeqLen)
      else
        ys ++ Seq.fill[Int](config.maxSeqLen - seq.length)(-1)
    })
    // transform to have the "label" column
    val gf = ef.withColumn("label", g(col("transitions")))    
    gf.select("label").show(10, false)
    val gfV = efV.withColumn("label", g(col("transitions")))
    (vocabMap, gf, gfV)
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
    val treebanks = Seq("atis", "eslspok", "ewt", "gum", "lines", "partut", "pud")
    val dfs = treebanks.map { name =>
      val path = s"dat/dep/UD_English/$name-${config.system}.jsonl"
      spark.read.json(path).filter(size(col("transitions")) >= 3)
    }
    val all = dfs.reduce((u, v) => u.union(v))
    logger.info("Number of samples = " + all.count)
    val Array(df, dfV) = all.randomSplit(Array(0.8, 0.2), seed = 150909)
    dfV.show(10)

    val (vocabMap, trainDF, validDF) = preprocess(df, dfV, config)
    // save the pretraining data to JSONL
    // println(s"vocabSize = $vocabSize")
    // trainDF.repartition(4).write.json("dat/dep/UD_English/asp-train")
    // validDF.repartition(1).write.json("dat/dep/UD_English/asp-valid")

    val model = createModel(config, vocabMap.size)
    model.summary()
    
    // train the BERT model
    train(model, config, trainDF, validDF)

    spark.stop()
  }
}
