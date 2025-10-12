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

object ClassifierType extends Enumeration {
  val MLR, MLP = Value
}

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
   * Empirically, we set the weight of SH to 0.1 (ten times smaller than the other transitions).
   */
  private def addWeightCol(df: DataFrame): DataFrame = {
    val f = udf((transition: String) => if (transition == "SH") 0.1 else 1.0)
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
    * Trains a classifier to predict transition of a given parsing config.
    *
    * @param modelPath
    * @param graphs
    * @param classifierType type of classifier (MLR, MLP)
    * @param hiddenLayers applicable only for MLP classifier
    * @return a pipeline model
    */
  def train(modelPath: String, graphs: List[Graph], classifierType: ClassifierType.Value, hiddenLayers: Array[Int]): PipelineModel = {
    val input = addWeightCol(createDF(graphs))
    input.write.mode(SaveMode.Overwrite).save("/tmp/tdp")
    val featureList = input.select("bof").collect().map(row => row.getString(0)).flatMap(s => s.split("\\s+"))
    val featureSet = featureList.toSet
    val featureCounter = featureList.groupBy(identity).mapValues(_.size).filter(p => p._2 >= config.minFrequency)
    if (config.verbose) {
      logger.info("#(distinctFeatures) = " + featureSet.size)
      logger.info(s"#(features with count >= ${config.minFrequency}) = " + featureCounter.size)
      input.show(10, false)
    }

    val labelIndexer = new StringIndexer().setInputCol("transition").setOutputCol("label").setHandleInvalid("skip")
    val tokenizer = new Tokenizer().setInputCol("bof").setOutputCol("tokens")
    val countVectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("features").setMinDF(config.minFrequency).setVocabSize(config.numFeatures)

    val model = classifierType match {
      case ClassifierType.MLR =>
        val mlr = new LogisticRegression().setMaxIter(config.iterations).setStandardization(false).setTol(1E-5).setWeightCol("weight")
        new Pipeline().setStages(Array(labelIndexer, tokenizer, countVectorizer, mlr)).fit(input)
      case ClassifierType.MLP =>
        val pipeline = new Pipeline().setStages(Array(labelIndexer, tokenizer, countVectorizer))
        val pipelineModel = pipeline.fit(input)
        val vocabSize = pipelineModel.stages(2).asInstanceOf[CountVectorizerModel].vocabulary.size
        val numLabels = input.select("transition").distinct().count().toInt
        val layers = Array[Int](vocabSize) ++ hiddenLayers ++ Array[Int](numLabels)
        val mlp = new MultilayerPerceptronClassifier().setLayers(layers).setMaxIter(config.iterations).setTol(1E-5).setBlockSize(config.batchSize)
        new Pipeline().setStages(Array(labelIndexer, tokenizer, countVectorizer, mlp)).fit(input)
    }

    // overwrite the trained pipeline
    model.write.overwrite().save(modelPath)
    // print some strings to debug the model
    if (config.verbose) {
      logger.info("#(labels) = " + model.stages(0).asInstanceOf[StringIndexerModel].labels.length)
      logger.info("#(vocabs) = " + model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary.size)
      classifierType match {
        case ClassifierType.MLR => logger.info(model.stages(3).asInstanceOf[LogisticRegressionModel].explainParams())
        case ClassifierType.MLP => logger.info(model.stages(3).asInstanceOf[MultilayerPerceptronClassificationModel].explainParams())
      }
    }
    model
  }

  /**
    * Extended version of [[train()]] method where super-tag vectors are concatenated.
    * @param modelPath
    * @param graphs
    * @param classifierType
    * @param hiddenLayers applicable only for MLP classifier
    * @param wordVectors
    * @param discrete                   
    * @return a model
    */
  def train(modelPath: String, graphs: List[Graph], classifierType: ClassifierType.Value, hiddenLayers: Array[Int],
            wordVectors: Map[String, Vector[Double]], discrete: Boolean = true): PipelineModel = {
    val input = addWeightCol(createDF(graphs, wordVectors, discrete))
    input.cache()
    val featureList = input.select("bof").collect().map(row => row.getString(0)).flatMap(s => s.split("\\s+"))
    val featureSet = featureList.toSet
    val featureCounter = featureList.groupBy(identity).mapValues(_.size).filter(p => p._2 >= config.minFrequency)
    if (config.verbose) {
      logger.info("#(distinctFeatures) = " + featureSet.size)
      logger.info(s"#(features with count >= ${config.numFeatures}) = " + featureCounter.size)
    }

    val labelIndexer = new StringIndexer().setInputCol("transition").setOutputCol("label").setHandleInvalid("skip")
    val tokenizer = new Tokenizer().setInputCol("bof").setOutputCol("tokens")
    val countVectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("f").setMinDF(config.minFrequency).setVocabSize(config.numFeatures)
    val vectorAssembler = new VectorAssembler().setInputCols(Array("f", "s", "q")).setOutputCol("features")

    val model = classifierType match {
      case ClassifierType.MLR => {
        val mlr = new LogisticRegression().setMaxIter(config.iterations).setStandardization(false).setTol(1E-5).setWeightCol("weight")
        new Pipeline().setStages(Array(labelIndexer, tokenizer, countVectorizer, vectorAssembler, mlr)).fit(input)
      }
      case ClassifierType.MLP => {
        val pipeline = new Pipeline().setStages(Array(labelIndexer, tokenizer, countVectorizer))
        val pipelineModel = pipeline.fit(input)
        val vocabSize = pipelineModel.stages(2).asInstanceOf[CountVectorizerModel].vocabulary.size
        val numLabels = input.select("transition").distinct().count().toInt
        val layers = Array[Int](vocabSize + distributedDimension * 2) ++ hiddenLayers ++ Array[Int](numLabels)
        val mlp = new MultilayerPerceptronClassifier().setLayers(layers).setMaxIter(config.iterations).setTol(1E-5).setBlockSize(64)
        new Pipeline().setStages(Array(labelIndexer, tokenizer, countVectorizer, vectorAssembler, mlp)).fit(input)
      }
    }

    // overwrite the trained pipeline
    model.write.overwrite().save(modelPath)
    // print some strings to debug the model
    if (config.verbose) {
      logger.info("#(labels) = " + model.stages(0).asInstanceOf[StringIndexerModel].labels.length)
      logger.info("#(vocabs) = " + model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary.size)
      classifierType match {
        case ClassifierType.MLR => logger.info(model.stages(4).asInstanceOf[LogisticRegressionModel].explainParams())
        case ClassifierType.MLP => logger.info(model.stages(4).asInstanceOf[MultilayerPerceptronClassificationModel].explainParams())
      }
    }
    model
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
      outputDF.select("transition", "label", "tokens", "features", "prediction").show(10, false)
      logger.info(model.stages(2).asInstanceOf[CountVectorizerModel].explainParams())
      val labels = metrics.labels
      labels.foreach(label => {
        val sb = new StringBuilder()
        sb.append(s"Precision($label) = " + metrics.precision(label) + ", ")
        sb.append(s"Recall($label) = " + metrics.recall(label) + ", ")
        sb.append(s"F($label) = " + metrics.fMeasure(label))
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
  }
}

object TransitionClassifier {
  val logger = LoggerFactory.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val parser = new OptionParser[ConfigTDP](getClass().getName()) {
      head(getClass().getName(), "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores, default is 8")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
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
      opt[Unit]('x', "extended").action((_, conf) => conf.copy(extended = true)).text("extended mode for English parsing")
      opt[Int]('s', "tag embedding size").action((x, conf) => conf.copy(tagEmbeddingSize = x)).text("tag embedding size 10/20/40")
    }
    parser.parse(args, ConfigTDP()) match {
      case Some(config) =>
        val spark = SparkSession.builder().appName(getClass.getName)
          .master(config.master)
          .config("spark.executor.cores", config.executorCores.toString)
          .config("spark.cores.max", config.totalCores.toString)
          .config("spark.executor.memory", config.executorMemory)
          .config("spark.driver.memory", config.driverMemory)
          .config("spark.shuffle.blockTransferService", "nio")
          .getOrCreate()
        val extended = config.extended
        val modelPath = Paths.get(config.modelPath, config.language, config.classifier).toString()
        val (trainingGraphs, developmentGraphs) = (
          GraphReader.read("dat/dep/UD_English-EWT/en_ewt-ud-train.conllu").filter(g => g.sentence.length >= 3), 
          GraphReader.read("dat/dep/UD_English-EWT/en_ewt-ud-test.conllu").filter(g => g.sentence.length >= 3)
        )
        val classifierType = config.classifier match {
          case "mlr" => ClassifierType.MLR
          case "mlp" => ClassifierType.MLP
        }
        // val ltagPath = "dat/dep/eng/tag/templates." + config.tagEmbeddingSize + ".txt"
        // val wordVectors = WordVectors.read(ltagPath)

        logger.info("#(trainingGraphs) = " + trainingGraphs.size)
        logger.info("#(developmentGraphs) = " + developmentGraphs.size)
        logger.info("modelPath = " + modelPath)
        logger.info("classifierName = " + config.classifier)

        val classifier = new TransitionClassifier(spark, config)
        val discrete = config.discrete

        config.mode match {
          case "eval" => {
            classifier.evalManual(modelPath, developmentGraphs)
            classifier.evalManual(modelPath, trainingGraphs)
            if (!extended) {
              classifier.eval(modelPath, developmentGraphs)
              classifier.eval(modelPath, trainingGraphs)
            } else {
              // classifier.eval(modelPath, developmentGraphs, wordVectors, discrete)
              // classifier.eval(modelPath, trainingGraphs, wordVectors, discrete)
            }
          }
          case "train" => {
            val hiddenLayersConfig = config.hiddenUnits
            val hiddenLayers = if (hiddenLayersConfig.isEmpty) Array[Int](); else hiddenLayersConfig.split("[,\\s]+").map(_.toInt)
            if (!extended)
              classifier.train(modelPath, trainingGraphs, classifierType, hiddenLayers)
            else {
              // logger.info("ltagPath = " + ltagPath)
              // logger.info("#(wordVectors) = " + wordVectors.size)
              // classifier.train(modelPath, trainingGraphs, classifierType, hiddenLayers, wordVectors, discrete)
            }
            if (!extended) {
              classifier.eval(modelPath, developmentGraphs)
              classifier.eval(modelPath, trainingGraphs)
            } else {
              // logger.info("ltagPath = " + ltagPath)
              // logger.info("#(wordVectors) = " + wordVectors.size)
              // classifier.eval(modelPath, developmentGraphs, wordVectors, discrete)
              // classifier.eval(modelPath, trainingGraphs, wordVectors, discrete)
            }
          }
          case "test" => {
            if (!extended) {
              val ds = classifier.createDF(developmentGraphs)
              ds.show(10, false)
            } else {
              // val ds = classifier.createDF(developmentGraphs, wordVectors, discrete)
              // ds.show(10, false)
            }
          }
        }
        spark.stop()
      case None =>
    }
  }
}

// Y: total cores, X: executor cores (Y/X will be the number of executors)

// bloop run -p con -m vlp.dep.TransitionClassifier -- -v -m train -Y 16 -X 8 (-Z 20gb)

// stack length of the parsing context (AS) on the dev. split of EWT.
// ---+-----+
// |  s|count|
// +---+-----+
// |  2|11985|
// |  3|10022|
// |  1| 8397|
// |  4| 7614|
// |  5| 4772|
// |  6| 2623|
// |  0| 1860|
// |  7| 1378|
// |  8|  690|
// |  9|  323|
// | 10|  145|
// | 11|   58|
// | 12|   18|
// | 13|    6|
// | 14|    2|
// | 15|    2|
// | 16|    2|
// | 17|    1|
// +---+-----+

// There are 49,898 samples.

// On the train split of the EWT.
// ???
// There are 406,252 samples. 