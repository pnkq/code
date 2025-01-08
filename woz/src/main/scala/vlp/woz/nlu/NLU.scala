package vlp.woz.nlu

import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.layers.{Bidirectional, Dense, Embedding, InputLayer, LSTM}
import com.intel.analytics.bigdl.dllib.nn.{ClassNLLCriterion, TimeDistributedCriterion}
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.{Adam, Trigger}
import com.intel.analytics.bigdl.dllib.utils.{Engine, Shape}
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation._
import scopt.OptionParser
import vlp.woz.act.Act

case class Span(
  actName: String,
  slot: String,
  value: String,
  start: Option[Long],
  end: Option[Long]
)

case class Element(
  dialogId: String,
  turnId: String,
  utterance: String,
  acts: Array[Act],
  spans: Array[Span]
)

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.optim.{AccuracyResult, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

/**
 * phuonglh@gamil.com
 *
 * @param paddingValue a padding value in the target sequence, default is -1f.
 * @param ev
 *
 * Note: 1-based label index for token classification
 */
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
        //      println(zs.mkString(", ") + " :=: " + ys.toArray().mkString(", ")) // DEBUG
        // filter the padded value (-1) in the target before perform matching with the output
        val c = ys.toArray().map(_.toInt).filter(e => e != paddingValue).zip(zs)
          .map(p => if (p._1 == p._2) 1 else 0)
        correct += c.sum
        count += c.length
      }
    new AccuracyResult(correct, count)
  }
  override def format(): String = "TimeDistributedTop1Accuracy"
}

/**
  * Reads dialog act data sets which are saved by [[vlp.woz.DialogReader]] and prepare 
  * data sets suitable for training token classification (sequence labeling) models.
  * 
  */
object NLU {

  private val pattern = """[?.,!\s]+"""
  private val numCores = Runtime.getRuntime.availableProcessors()

  /**
    * Given an utterance and its associated non-empty spans, tokenize the utterance 
    * into tokens and their corresponding slot labels (B/I/O).
    * @param utterance an utterance
    * @param acts a set of acts
    * @param spans an array of spans
    * @return a sequence of tuples.
    */
  def tokenize(utterance: String, acts: Array[Act], spans: Array[Span]): Seq[(Int, Array[(String, String)])] = {
    if (spans.length > 0) {
      val intervals: Array[(Int, Int)] = spans.map { span => (span.start.get.toInt, span.end.get.toInt) }
      val (a, b) = (intervals.head._1, intervals.last._2)
      // build intervals that need to be tokenized
      val js = new collection.mutable.ArrayBuffer[(Int, Int)](intervals.length + 1)
      if (a > 0) js.append((0, a))
      for (j <- 0 until intervals.length - 1) {
        // there exists the cases of two similar intervals with different slots. We deliberately ignore those cases for now.
        if (intervals(j)._2 < intervals(j+1)._1) {
          js.append((intervals(j)._2, intervals(j+1)._1))
        }
      }
      if (b < utterance.length) js.append((b, utterance.length))
      // build results
      val ss = new collection.mutable.ArrayBuffer[(Int, Array[(String, String)])](intervals.length*2)
      for (j <- intervals.indices) {
        val p = intervals(j)
        val slot = spans(j).slot
        val value = utterance.subSequence(p._1, p._2).toString.trim()
        val tokens = value.split(pattern)
        val labels = s"B-${slot.toUpperCase()}" +: Array.fill[String](tokens.size-1)(s"I-${slot.toUpperCase()}")
        ss.append((p._1, tokens.zip(labels)))
      }
      js.foreach { p => 
        val text = utterance.subSequence(p._1, p._2).toString.trim()
        if (text.nonEmpty) {
          val tokens = text.split(pattern).filter(_.nonEmpty)
          ss.append((p._1, tokens.zip(Array.fill[String](tokens.length)("O"))))
        }
      }
      // the start indices are used to sort the sequence of triples
      ss.sortBy(_._1)
    } else {
      // there is no slots, extract only O-labeled tokens
      val tokens = utterance.split(pattern).filter(_.nonEmpty)
      Seq((0, tokens.zip(Array.fill[String](tokens.length)("O"))))
    }
  }

  private def extractActNames(acts: Array[Act]): Array[String] = {
    acts.map(act => act.name.toUpperCase()).distinct.sorted
  }

  private val f = udf((utterance: String, acts: Array[Act], spans: Array[Span]) => tokenize(utterance, acts, spans))
  // extract tokens
  private val f1 = udf((seq: Seq[(Int, Array[(String, String)])]) => seq.flatMap(_._2.map(_._1)))
  // extract slots
  private val f2 = udf((seq: Seq[(Int, Array[(String, String)])]) => seq.flatMap(_._2.map(_._2)))
  // extract actNames
  private val g = udf((acts: Array[Act]) => extractActNames(acts))

  /**
    * Reads a data set and creates a df of columns (utterance, tokenSequence, slotSequence, actNameSequence), where
    * <ol>
    * <li>utterance: String, is a original text</li>
    * <li>tokenSequence: Seq[String], is a sequence of tokens from utterance</li>
    * <li>slotSequence: Seq[String], is a sequence of slot names (entity types, in the form of B/I/O)</li>
    * <li>actNameSequence: Seq[String], is a sequence of act names, which is typically 1 or 2 act names.
    * </ol>
    *
    * @param spark a spark session
    * @param path a path to a split data
    */
  def transformActs(spark: SparkSession, path: String): DataFrame = {
    import spark.implicits._
    val af = spark.read.json(path).as[Element]
    println("Number of rows = " + af.count())
    // filter for rows with non-empty spans
    // val bf = af.filter(size(col("spans")) > 0)
    val cf = af.withColumn("seq", f(col("utterance"), col("acts"), col("spans")))
      .withColumn("tokens", f1(col("seq")))
      .withColumn("slots", f2(col("seq")))
      .withColumn("actNames", g(col("acts")))
    cf.select("dialogId", "turnId", "utterance", "tokens", "slots", "actNames")
  }

  private def saveDatasets(spark: SparkSession): Unit = {
    val splits = Array("train", "dev", "test")
    splits.foreach { split => 
      val df = transformActs(spark, s"dat/woz/act/$split")
      df.repartition(1).write.mode("overwrite").json(s"dat/woz/nlu/$split")
    }
  }

  private def preprocess(df: DataFrame, savePath: String = ""): PipelineModel = {
    val vectorizerToken = new CountVectorizer().setInputCol("tokens").setOutputCol("tokenVec")
    val vectorizerSlot = new CountVectorizer().setInputCol("slots").setOutputCol("slotVec")
    val vectorizerAct = new CountVectorizer().setInputCol("actNames").setOutputCol("actVec")
    val pipeline = new Pipeline().setStages(Array(vectorizerToken, vectorizerSlot, vectorizerAct))
    val model = pipeline.fit(df)
    if (savePath.nonEmpty) model.write.save(savePath)
    model
  }

  private def createEncoder(numTokens: Int, numEntities: Int, numActs: Int, config: ConfigNLU): Sequential[Float] = {
    val sequential = Sequential[Float]()
    sequential.add(InputLayer[Float](inputShape = Shape(config.maxSeqLen)))
    sequential.add(Embedding[Float](inputDim = numTokens, outputDim = config.embeddingSize))
    for (j <- 0 until config.numLayers)
      sequential.add(Bidirectional[Float](LSTM[Float](outputDim = config.recurrentSize, returnSequences = true)))
    sequential.add(Dense[Float](config.hiddenSize))
    sequential.add(Dense[Float](numEntities, activation = "softmax"))
    sequential
  }

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[ConfigNLU](getClass.getName) {
      head(getClass.getName, "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 16g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either {eval, train, predict}")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('j', "numLayers").action((x, conf) => conf.copy(numLayers = x)).text("number of RNN layers or Transformer blocks")
      opt[Int]('h', "hiddenSize").action((x, conf) => conf.copy(hiddenSize = x)).text("encoder hidden size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type")
    }
    opts.parse(args, ConfigNLU()) match {
      case Some(config) =>
        val conf = new SparkConf().setAppName(getClass.getName).setMaster(config.master)
          .set("spark.executor.cores", config.executorCores.toString)
          .set("spark.cores.max", config.totalCores.toString)
          .set("spark.executor.memory", config.executorMemory)
          .set("spark.driver.memory", config.driverMemory)
        val sc = new SparkContext(conf)
        Engine.init
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
        sc.setLogLevel("WARN")

        val basePath = "dat/woz/nlu"
        config.mode match {
          case "init" =>
            saveDatasets(spark)
            val df = spark.read.json("dat/woz/nlu/train")
            preprocess(df, s"$basePath/pre")
          case "train" =>
            val preprocessor = PipelineModel.load(s"$basePath/pre")
            val vocab = preprocessor.stages(0).asInstanceOf[CountVectorizerModel].vocabulary
            val vocabDict = vocab.zipWithIndex.map(p => (p._1, p._2)).toMap
            val entities = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
            val entityDict = entities.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
            val acts = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary

            val sequencerTokens = new Sequencer(vocabDict, config.maxSeqLen, 0f).setInputCol("tokens").setOutputCol("tokenIdx")
            val sequencerEntities = new Sequencer(entityDict, config.maxSeqLen, -1f).setInputCol("slots").setOutputCol("slotIdx")

            val (train, dev) = (spark.read.json("dat/woz/nlu/train"), spark.read.json("dat/woz/nlu/dev"))
            val (uf, vf) = (sequencerTokens.transform(sequencerEntities.transform(train)), sequencerTokens.transform(sequencerEntities.transform(dev)))
            uf.select("tokenIdx", "slotIdx").show(false)
            val encoder = createEncoder(vocab.length, entities.length, acts.length, config)
            encoder.summary()

            val (featureSize, labelSize) = (Array(config.maxSeqLen), Array(config.maxSeqLen))
            // should set the sizeAverage=false in ClassNLLCriterion
            val criterion = ClassNLLCriterion[Float](sizeAverage = false, paddingValue = -1)
            val estimator = NNEstimator(encoder, TimeDistributedCriterion(criterion, sizeAverage = true), featureSize, labelSize)
            val trainingSummary = TrainSummary(appName = config.modelType, logDir = "sum/woz/")
            val validationSummary = ValidationSummary(appName = config.modelType, logDir = "sum/woz/")
            val batchSize = if (config.batchSize % numCores != 0) numCores * 4; else config.batchSize
            estimator.setLabelCol("slotIdx").setFeaturesCol("tokenIdx")
              .setBatchSize(batchSize)
              .setOptimMethod(new Adam[Float](1E-4))
              .setMaxEpoch(config.epochs)
              .setTrainSummary(trainingSummary)
              .setValidationSummary(validationSummary)
              .setValidation(Trigger.everyEpoch, vf, Array(new TimeDistributedTop1Accuracy(paddingValue = -1)), batchSize)
            estimator.fit(vf) // uf in server

            encoder.saveModel(s"bin/woz-${config.modelType}.bigdl", overWrite = true)

          case "eval" =>
        }
        spark.stop()
      case None =>
    }
  }
}
