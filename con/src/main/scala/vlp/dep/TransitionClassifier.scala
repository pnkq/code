package vlp.dep

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession, SaveMode}
import org.apache.spark.storage.StorageLevel
import org.slf4j.LoggerFactory
import scopt.OptionParser
import java.nio.file.Paths

import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.{Top1Accuracy, Trigger}
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}


case class Phrase(tokens: Seq[String])

/**
  * Created by phuonglh on 6/24/17.
  *
  * A transition classifier.
  *
  */
class TransitionClassifier(spark: SparkSession, config: ConfigTDP) {
  val logger = LoggerFactory.getLogger(getClass.getName)
  val featureExtractor = new FeatureExtractor(false, false)
  val oracle = new OracleAS(featureExtractor) // arc-standard oracle
  val distributedDimension = 40

  /**
    * Creates a Spark data set of parsing contexts from a list of dependency graphs using an oracle.
    *
    * @param graphs a list of manually-annotated dependency graphs.
    * @return a data frame of parsing contexts.
    */
  private def createDF(graphs: List[Graph]): DataFrame = {
    val contexts = oracle.decode(graphs)
    if (config.verbose) logger.info("#(contexts) = " + contexts.length)
    spark.createDataFrame(contexts)
  }

  /**
   * The createDF(graphs) above creates a data frame of 3 columns (id, bof, transition).
   * This method add a new column which specifies the weights for each label "transition". 
   * The SH transition is usually the most frequent label, so it should have a small weight. 
   * 
   */
  private def addWeightCol(df: DataFrame): DataFrame = {
    import spark.implicits._
    val size = df.count
    val weightMap = df.groupBy("transition").count().map { row => 
      val label = row.getString(0)
      val weight = row.getLong(1).toDouble/size
      (label, weight)
    }.collect().toMap

    val f = udf((transition: String) => weightMap(transition))
    df.withColumn("weight", f(col("transition")))
  }
  
  /**
    * Creates an extended data frame with distributed word representations concatenated to feature vectors.
    * @param graphs
    * @param wordVectors
    * @param withDiscrete use continuous features together with their discrete features                    
    * @return a DataFrame
    */
  private def createDF(graphs: List[Graph], wordVectors: Map[String, Vector[Double]], withDiscrete: Boolean = true): DataFrame = {
    def suffix(f: String): String = {
      val idx = f.indexOf(':')
      if (idx >= 0) f.substring(idx + 1); else ""
    }
    val seqAsVector = udf((xs: Seq[Double]) => Vectors.dense(xs.toArray))

    val oracle = new OracleAS(featureExtractor)
    val contexts = oracle.decode(graphs)
    val zero = Vector.fill(distributedDimension)(0d)
    val extendedContexts = contexts.map(c => {
      val fs = c.bof.split("\\s+")
      val s = fs.filter(f => f.startsWith("sts0:"))
      val ws0 = suffix(if (s.nonEmpty) s.head else "UNK")
      val q = fs.filter(f => f.startsWith("stq0:"))
      val wq0 = suffix(if (q.nonEmpty) q.head else "UNK")
      if (withDiscrete)
        ExtendedContext(c.id, c.bof, c.transition, wordVectors.getOrElse(ws0, zero), wordVectors.getOrElse(wq0, zero))
      else {
        val ff = c.bof.split("\\s+").filterNot(f => f.startsWith("sts0:") || f.startsWith("stq0:"))
        ExtendedContext(c.id, ff.mkString(" "), c.transition, wordVectors.getOrElse(ws0, zero), wordVectors.getOrElse(wq0, zero))
      }
    })
    if (config.verbose) logger.info("#(contexts) = " + contexts.length)
    import spark.implicits._
    val ds = spark.createDataFrame(extendedContexts).as[ExtendedContext]
    ds.withColumn("s", seqAsVector(col("s"))).withColumn("q", seqAsVector(col("q"))).toDF()
  }

  /**
    * Trains a classifier to predict transition of a given parsing config. This method uses only discrete BoF. 
    *
    * @param modelPath
    * @param graphs
    * @param hiddenLayers applicable only for MLP classifier
    * @return a pipeline model
    */
  def train(modelPath: String, graphs: List[Graph], hiddenLayers: Array[Int]): PipelineModel = {
    val input = addWeightCol(createDF(graphs))
    // input.write.mode(SaveMode.Overwrite).save("/tmp/tdp")
    val labelIndexer = new StringIndexer().setInputCol("transition").setOutputCol("label").setHandleInvalid("skip")
    val tokenizer = new Tokenizer().setInputCol("bof").setOutputCol("tokens")
    val countVectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("features").setMinDF(config.minFrequency).setVocabSize(config.numFeatures)

    val model = if (hiddenLayers.isEmpty) {
      val mlr = new LogisticRegression().setMaxIter(config.iterations).setStandardization(false).setTol(1E-5).setWeightCol("weight")
      new Pipeline().setStages(Array(labelIndexer, tokenizer, countVectorizer, mlr)).fit(input)
    } else {
      val pipeline = new Pipeline().setStages(Array(labelIndexer, tokenizer, countVectorizer))
      val pipelineModel = pipeline.fit(input)
      val vocabSize = pipelineModel.stages(2).asInstanceOf[CountVectorizerModel].vocabulary.size
      val numLabels = input.select("transition").distinct().count().toInt
      val layers = Array[Int](vocabSize) ++ hiddenLayers ++ Array[Int](numLabels)
      val mlp = new MultilayerPerceptronClassifier().setLayers(layers).setMaxIter(config.iterations).setTol(1E-5).setBlockSize(config.batchSize)
      new Pipeline().setStages(Array(labelIndexer, tokenizer, countVectorizer, mlp)).fit(input)
    }

    // overwrite the trained pipeline model
    model.write.overwrite().save(modelPath)
    // print some strings to debug the model
    if (config.verbose) {
      logger.info("  #(labels) = " + model.stages(0).asInstanceOf[StringIndexerModel].labels.length)
      logger.info("#(features) = " + model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary.size)
      val modelSt = if (hiddenLayers.isEmpty) {
        model.stages(3).asInstanceOf[LogisticRegressionModel].explainParams()
      } else {
        model.stages(3).asInstanceOf[MultilayerPerceptronClassificationModel].explainParams()
      }
      logger.info(modelSt)
    }
    model
  }

  /**
    * Extended version of [[train()]] method where super-tag vectors are concatenated.
    * @param modelPath
    * @param graphs
    * @param hiddenLayers 
    * @param wordVectors
    * @param discrete                   
    * @return a model
    */
  def train(modelPath: String, graphs: List[Graph], hiddenLayers: Array[Int], wordVectors: Map[String, Vector[Double]], discrete: Boolean = true): PipelineModel = {
    val input = addWeightCol(createDF(graphs, wordVectors, discrete))
    input.cache()
    val labelIndexer = new StringIndexer().setInputCol("transition").setOutputCol("label").setHandleInvalid("skip")
    val tokenizer = new Tokenizer().setInputCol("bof").setOutputCol("tokens")
    val countVectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("f").setMinDF(config.minFrequency).setVocabSize(config.numFeatures)
    val vectorAssembler = new VectorAssembler().setInputCols(Array("f", "s", "q")).setOutputCol("features")

    val model = if (hiddenLayers.isEmpty) {
      val mlr = new LogisticRegression().setMaxIter(config.iterations).setStandardization(false).setTol(1E-5).setWeightCol("weight")
      new Pipeline().setStages(Array(labelIndexer, tokenizer, countVectorizer, vectorAssembler, mlr)).fit(input)
    } else {
      val pipeline = new Pipeline().setStages(Array(labelIndexer, tokenizer, countVectorizer))
      val pipelineModel = pipeline.fit(input)
      val vocabSize = pipelineModel.stages(2).asInstanceOf[CountVectorizerModel].vocabulary.size
      val numLabels = input.select("transition").distinct().count().toInt
      val layers = Array[Int](vocabSize + distributedDimension * 2) ++ hiddenLayers ++ Array[Int](numLabels)
      val mlp = new MultilayerPerceptronClassifier().setLayers(layers).setMaxIter(config.iterations).setTol(1E-5).setBlockSize(64)
      new Pipeline().setStages(Array(labelIndexer, tokenizer, countVectorizer, vectorAssembler, mlp)).fit(input)
    }

    // overwrite the trained pipeline model
    model.write.overwrite().save(modelPath)
    // print some strings to debug the model
    if (config.verbose) {
      logger.info("  #(labels) = " + model.stages(0).asInstanceOf[StringIndexerModel].labels.length)
      logger.info("#(features) = " + model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary.size)
      val modelSt = if (hiddenLayers.isEmpty) {
        model.stages(4).asInstanceOf[LogisticRegressionModel].explainParams()
      } else {
        model.stages(4).asInstanceOf[MultilayerPerceptronClassificationModel].explainParams()
      }
      logger.info(modelSt)
    }
    model
  }

  /**
    * Trains a classifier to predict transition of a given parsing config. This method uses an LSTM to extract features 
    * from the stack and the top token on the queue. 
    *
    * @param config 
    * @param graphs
    * @return a pipeline model
    */
  def train(config: ConfigTDP, graphs: List[Graph]) = {
    val input = addWeightCol(createDF(graphs)).sample(0.2)
    // bof preprocessing pipeline
    val labelIndexer = new StringIndexer().setInputCol("transition").setOutputCol("label").setHandleInvalid("skip")
    val tokenizer = new Tokenizer().setInputCol("bof").setOutputCol("tokens")
    val countVectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("features").setMinDF(config.minFrequency).setVocabSize(config.numFeatures)
    logger.info("Fitting the preprocessor pipeline. Please wait...")
    val preprocessor = new Pipeline().setStages(Array(labelIndexer, tokenizer, countVectorizer)).fit(input.select("bof", "transition"))
    val labelSize = preprocessor.stages(0).asInstanceOf[StringIndexerModel].labels.length
    logger.info("  #(labels) = " + labelSize)
    // the sentence data frame for LSTM sequence
    val sentences = graphs.map(graph => graph.sentence.tokens.map(token => token.word.toLowerCase).toSeq).map(s => Phrase(s))
    val df = spark.createDataFrame(sentences)
    df.show(false)
    val vectorizer = new CountVectorizer().setInputCol("tokens")
    logger.info("Fitting the word vectorizer. Please wait...")
    val vectorizerModel = vectorizer.fit(df)

    // get the vocabulary and convert it to a map (word -> id), reserve 0 for UNK token
    val vocab = vectorizerModel.vocabulary.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
    logger.info(s"#(words) = ${vocab.size}")
    
    // create a udf to extract a seq of tokens from the stack and the queue, left-pad the seq with -1. 
    // set the maximum sequence of 10 (empirically found!)
    val maxSeqLen = 10
    val f = udf((stack: Seq[String], queue: Seq[String]) => {
      val xs = stack :+ queue.head
      val is = xs.map(x => vocab.getOrElse(x, 0))
      if (is.length < maxSeqLen) Seq.fill(maxSeqLen)(-1) ++ is else is.take(maxSeqLen)
    })
    val ef = input.withColumn("seq", f(col("stack"), col("queue")))
    ef.select("transition", "label", "seq").show(false)

    val sequential = Sequential[Float]()
    val masking = Masking[Float](-1, inputShape = Shape(maxSeqLen)).setName("masking")
    sequential.add(masking)
    val embedding = Embedding[Float](vocab.size, config.featureEmbeddingSize, paddingValue = -1).setName("embedding")
    sequential.add(embedding)
    val lstm = LSTM[Float](36).setName("lstm")
    sequential.add(lstm)
    val dense = Dense[Float](labelSize, activation = "softmax")
    sequential.add(dense)
    sequential.summary()

    // overwrite the trained preprocessor
    val modelPath = Paths.get(config.modelPath, config.language, config.classifier).toString()
    preprocessor.write.overwrite().save(modelPath)
    // print some strings to debug the model
    if (config.verbose) {
      logger.info("  #(labels) = " + preprocessor.stages(0).asInstanceOf[StringIndexerModel].labels.length)
      logger.info("#(features) = " + preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary.size)
    }
    val criterion = ClassNLLCriterion[Float](sizeAverage = false, logProbAsInput = false)
    val estimator = NNEstimator(sequential, criterion, Array(maxSeqLen), Array(1))
    val trainingSummary = TrainSummary(appName = "rnn", logDir = "sum/tdp/")
    val validationSummary = ValidationSummary(appName = "rnn", logDir = "sum/tdp/")
    val numCores = Runtime.getRuntime.availableProcessors()
    val batchSize = numCores * 4
    estimator.setLabelCol("label").setFeaturesCol("seq")
      .setBatchSize(batchSize)
      .setOptimMethod(new Adam(1E-3))
      .setMaxEpoch(40)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, ef, Array(new Top1Accuracy[Float]()), batchSize)
    estimator.fit(ef)
    sequential.saveModel(Paths.get(config.modelPath, config.language).toString() + "/rnn.bigdl", overWrite = true)
  }
  /**
    * Evaluates the accuracy of the transition classifier, including overall accuracy,
    * precision, recall and f-measure ratios for each transition label.
    *
    * @param modelPath trained model path
    * @param graphs    a list of dependency graphs
    * @param discrete                 
    */
  def eval(modelPath: String, graphs: List[Graph], wordVectors: Map[String, Vector[Double]] = null, discrete: Boolean = true): Unit = {
    val model = PipelineModel.load(modelPath)
    val inputDF = if (wordVectors == null) createDF(graphs) ; else createDF(graphs, wordVectors, discrete)
    import spark.implicits._
    val outputDF = model.transform(inputDF)
    val predictionAndLabels = outputDF.select("label", "prediction").map(row => (row.getDouble(0), row.getDouble(1))).rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val accuracy = metrics.accuracy
    logger.info(s"Accuracy = $accuracy")
    if (config.verbose) {
      outputDF.select("transition", "label", "tokens", "prediction").show(10, false)
      // logger.info(model.stages(2).asInstanceOf[CountVectorizerModel].explainParams())
      val labels = metrics.labels
      labels.foreach(label => {
        val sb = new StringBuilder()
        sb.append(s"$label: P = " + "%.4f".format(metrics.precision(label)) + ", ")
        sb.append(s"R = " + "%.4f".format(metrics.recall(label)) + ", ")
        sb.append(s"F = " + "%.4f".format(metrics.fMeasure(label)) + ", ")
        sb.append(s"A = " + "%.4f".format(metrics.accuracy))
        logger.info(sb.toString)
      })
    }
  }

  /**
    * Loads a classifier from the trained pipeline and manually computes the prediction
    * instead of relying on the transform() method of the SparkML framework.
    * @param modelPath
    * @param graphs
    */
  def evalManual(modelPath: String, graphs: List[Graph]): Unit = {
    val model = PipelineModel.load(modelPath)
    val oracle = new OracleAS(featureExtractor)
    val contexts = oracle.decode(graphs)
    val classifier = if (modelPath.endsWith("mlp")) new MLP(spark, model, featureExtractor) else new MLR(spark, model, featureExtractor)
    val output = contexts.map(c => (c.transition, classifier.predict(c.bof).head))
    val corrects = output.filter(p => p._1 == p._2).size
    logger.info(classifier.info())
    logger.info("#(corrects) = " + corrects + ", accuracy = " + (corrects.toDouble / contexts.size))
    val nonSH = output.filter(p => p._1 != "SH")
    val correctsNonSH = nonSH.filter(p => p._1 == p._2).size
    logger.info("#(correctsNonSH) = " + correctsNonSH + ", accuracy = " + (correctsNonSH.toDouble / nonSH.size))
  }
}

object TransitionClassifier {
  val logger = LoggerFactory.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val parser = new OptionParser[ConfigTDP](getClass().getName()) {
      head(getClass().getName(), "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 8g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/test")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'bin/tdp/'")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode")
      opt[String]('c', "classifier").action((x, conf) => conf.copy(classifier = x)).text("classifier, either mlr or mlp")
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language, either vie or eng")
      opt[String]('h', "hiddenUnits").action((x, conf) => conf.copy(hiddenUnits = x)).text("hidden units of MLP")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('u', "numFeatures").action((x, conf) => conf.copy(numFeatures = x)).text("number of features")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('e', "feature embedding size").action((x, conf) => conf.copy(featureEmbeddingSize = x)).text("feature embedding size 10/20/40")
    }
    parser.parse(args, ConfigTDP()) match {
      case Some(config) =>
        val spark = SparkSession.builder().appName(getClass.getName)
          .master(config.master)
          .config("spark.driver.host", "localhost")
          .config("spark.driver.memory", config.driverMemory)
          .config("spark.shuffle.blockTransferService", "nio")
          .getOrCreate()
        val modelPath = Paths.get(config.modelPath, config.language, config.classifier).toString()
        val (trainingGraphs, developmentGraphs) = (
          GraphReader.read("dat/dep/UD_English-EWT/en_ewt-ud-dev.conllu").filter(g => g.sentence.length >= 3), 
          GraphReader.read("dat/dep/UD_English-EWT/en_ewt-ud-test.conllu").filter(g => g.sentence.length >= 3)
        )

        logger.info("#(trainingGraphs) = " + trainingGraphs.size)
        logger.info("#(developmentGraphs) = " + developmentGraphs.size)
        logger.info("modelPath = " + modelPath)
        logger.info("classifier = " + config.classifier)

        val classifier = new TransitionClassifier(spark, config)

        config.mode match {
          case "eval" => {
            classifier.eval(modelPath, developmentGraphs)
            classifier.evalManual(modelPath, developmentGraphs)
            classifier.eval(modelPath, trainingGraphs)
            classifier.evalManual(modelPath, trainingGraphs)
          }
          case "train" => {
            config.classifier match {
              case "rnn" => {
                Engine.init
                classifier.train(config, trainingGraphs)                
              }
              case _ =>
                val hiddenLayersConfig = config.hiddenUnits
                val hiddenLayers = if (hiddenLayersConfig.isEmpty) Array[Int](); else hiddenLayersConfig.split("[,\\s]+").map(_.toInt)
                classifier.train(modelPath, trainingGraphs, hiddenLayers)
            }
          }
          case "test" => 
        }
        spark.stop()
      case None =>
    }
  }
}

