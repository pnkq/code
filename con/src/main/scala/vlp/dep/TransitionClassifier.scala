package vlp.dep

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession, SaveMode}
import org.apache.spark.storage.StorageLevel
import org.slf4j.LoggerFactory
import scopt.OptionParser
import java.nio.file.Paths

import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.tensor.{Tensor, DenseTensor, SparseTensor}
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.models.{Model, Models}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.{Top1Accuracy, Trigger}
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import scala.util.Try


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
    val weightMap = df.groupBy("transition").count().map { row => 
      val label = row.getString(0)
      val weight = 10/row.getLong(1).toDouble
      (label, weight)
    }.collect().toMap

    val f = udf((transition: String) => weightMap(transition))
    df.withColumn("weight", f(col("transition")))
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

  // create a udf to increase label index from 0-based to 1-based
  val g = udf((k: Double) => k + 1)

  /**
  * Trains a classifier to predict transition given a parsing config. This is an enhanced version of [[trainC]] method. 
  * A pretrained transition context vector is appended as an additional input feature vector to predict the next sentence.
  * 
  */
  def trainD(config: ConfigTDP, graphs: List[Graph], devGraphs: List[Graph]) = {
    val af = addWeightCol(createDF(graphs))
    val bf = createDF(devGraphs) // no need to add weights for test df
    // bof preprocessing pipeline
    val labelIndexer = new StringIndexer().setInputCol("transition").setOutputCol("label").setHandleInvalid("skip")
    val tokenizer = new Tokenizer().setInputCol("bof").setOutputCol("tokens")
    val bofVectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("vec").setMinDF(config.minFrequency).setVocabSize(config.numFeatures)
    val wordVectorizer = new CountVectorizer().setInputCol("words")
    val tagVectorizer = new CountVectorizer().setInputCol("tags")
    logger.info("Fitting the preprocessor pipeline. Please wait...")
    val preprocessor = new Pipeline().setStages(Array(labelIndexer, tokenizer, bofVectorizer, wordVectorizer, tagVectorizer))
      .fit(af.select("transition", "bof", "words", "tags"))
    val ef = preprocessor.transform(af)
    val ff = preprocessor.transform(bf)

    val labels = preprocessor.stages(0).asInstanceOf[StringIndexerModel].labels
    val labelSize = labels.length
    logger.info("#(labels) = " + labelSize)
    val bof = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
    logger.info("   #(bof) = " + bof.size)
    val vocabulary = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    logger.info(" #(words) = " + vocabulary.size)
    val vocabularyT = preprocessor.stages(4).asInstanceOf[CountVectorizerModel].vocabulary
    logger.info("  #(tags) = " + vocabularyT.size)

    // get the vocabulary and convert it to a map (word -> id), reserve an additional 0 for UNK word
    val vocab = vocabulary.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
    // get the vocabularyT and convert it to a map (tag -> id), reserve an additional 0 for UNK tag
    val vocabT = vocabularyT.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
    
    // create a udf to extract a seq of word ids and a sequence of tag ids from the stack and the queue, left-pad the seq with -1. 
    // set the maximum sequence of 10 (empirically found!)
    val maxSeqLen = 10
    val f = udf((stack: Seq[String], queue: Seq[String], words: Seq[String], tags: Seq[String]) => {
      val xs = queue.head +: stack
      val is = xs.map(x => vocab.getOrElse(words(x.toInt), 0))
      val js = xs.map(x => vocabT.getOrElse(tags(x.toInt), 0))
      val seqW = if (is.length < maxSeqLen) Seq.fill(maxSeqLen-is.length)(-1) ++ is else is.take(maxSeqLen)
      val seqT = if (js.length < maxSeqLen) Seq.fill(maxSeqLen-js.length)(-1) ++ js else js.take(maxSeqLen)
      seqW ++ seqT
    })

    val uf = ef.withColumn("seq", f(col("stack"), col("queue"), col("words"), col("tags"))).withColumn("y", g(col("label")))
    val vf = ff.withColumn("seq", f(col("stack"), col("queue"), col("words"), col("tags"))).withColumn("y", g(col("label")))

    uf.select("vec", "transition", "y", "weight", "seq").show(false)
    // prepare input data frame for multi-input BigDL model: featuresCol = Seq(vec, seq), labelCol=y
    import spark.implicits._
    import org.apache.spark.sql.types._
    
    val ufB = uf.select("vec", "seq", "y").map { row => 
      val x1 = row.getAs[Vector](0).toArray.map(_.toFloat)
      val x2 = row.getAs[Seq[Int]](1).toArray.map(_.toFloat)
      val y = row.getAs[Double](2).toFloat
      (x1 ++ x2, y)
    }.toDF("features", "y")
    val vfB = vf.select("vec", "seq", "y").map { row => 
      val x1 = row.getAs[Vector](0).toArray.map(_.toFloat)
      val x2 = row.getAs[Seq[Int]](1).toArray.map(_.toFloat)
      val y = row.getAs[Double](2).toFloat
      (x1 ++ x2, y)
    }.toDF("features", "y")

    ufB.show(5)
    ufB.printSchema
    
    // create a BigDL model
    val input1 = Input[Float](inputShape = Shape(bof.size))  // bof
    val input2 = Input[Float](inputShape = Shape(maxSeqLen)) // seqW
    val input3 = Input[Float](inputShape = Shape(maxSeqLen)) // seqT
    val input4 = Input[Float](inputShape = Shape(config.transitionEmbeddingSize)) // contextualized, pretrained transition vector 
    val dense1 = Dense[Float](config.featureEmbeddingSize).setName("dense1").inputs(input1)
    val maskingW = Masking[Float](-1).setName("maskingW").inputs(input2)
    val embeddingW = Embedding[Float](vocab.size + 1, config.wordEmbeddingSize, paddingValue = -1).setName("embeddingW").inputs(maskingW)
    val maskingT = Masking[Float](-1).setName("maskingT").inputs(input3)
    val embeddingT = Embedding[Float](vocabT.size + 1, config.tagEmbeddingSize, paddingValue = -1).setName("embeddingT").inputs(maskingT)
    // concat word embeddings before passing to an LSTM
    val mergeWT = Merge.merge(inputs = List(embeddingW, embeddingT), mode = "concat")
    val lstm = LSTM[Float](config.recurrentSize).setName("lstm").inputs(mergeWT)
    // merge three branches, including the pretrained contextualized transition vector 
    val merge = Merge.merge(inputs = List(dense1, lstm, input4), mode = "concat")
    val output = Dense[Float](labelSize, activation = "softmax").setName("output").inputs(merge)
    val model = Model[Float](Array(input1, input2, input3, input4), output)
    model.summary()

    // overwrite the trained preprocessor
    val modelPath = Paths.get(config.modelPath, config.language, config.classifier).toString()
    preprocessor.write.overwrite().save(modelPath)
    // prepare a label weight tensor
    val weights = Tensor[Float](labelSize)
    val labelMap = uf.select("y", "weight").distinct.collect()
      .map(row => (row.getDouble(0).toInt, row.getDouble(1).toFloat)).toMap
    for (j <- labelMap.keys)
      weights(j) = labelMap(j)
    logger.info(labelMap.toString)

    val criterion = ClassNLLCriterion[Float](weights = weights, sizeAverage = false, logProbAsInput = false)
    val estimator = NNEstimator(model, criterion, Array(Array(bof.size), Array(maxSeqLen), Array(maxSeqLen), Array(config.transitionEmbeddingSize)), Array(1))
    val trainingSummary = TrainSummary(appName = "rnnD", logDir = "sum/tdp/")
    val validationSummary = ValidationSummary(appName = "rnnD", logDir = "sum/tdp/")
    val numCores = Runtime.getRuntime.availableProcessors()
    val batchSize = numCores * 16
    estimator.setLabelCol("y").setFeaturesCol("features")
      .setBatchSize(batchSize)
      .setOptimMethod(new Adam(1E-4))
      .setMaxEpoch(config.iterations)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, vfB, Array(new Top1Accuracy[Float]()), batchSize)
    estimator.fit(ufB)
    model.saveModel(Paths.get(config.modelPath, config.language).toString() + "/rnnD.bigdl", overWrite = true)
  }

  /**
    * Trains a classifier to predict transition given a parsing config. This method uses an LSTM to extract features 
    * from the stack and the top token on the queue. It also uses the bof features for prediction as in the MLR or MLP model. 
    * It use sequence of PoS tags as additional features, in addition to similar features trainB() method. 
    *
    * @param config 
    * @param graphs
    * @param devGraphs
    * @return a pipeline model
    */
  def trainC(config: ConfigTDP, graphs: List[Graph], devGraphs: List[Graph]) = {
    val af = addWeightCol(createDF(graphs))
    val bf = createDF(devGraphs) // no need to add weights for test df
    // bof preprocessing pipeline
    val labelIndexer = new StringIndexer().setInputCol("transition").setOutputCol("label").setHandleInvalid("skip")
    val tokenizer = new Tokenizer().setInputCol("bof").setOutputCol("tokens")
    val bofVectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("vec").setMinDF(config.minFrequency).setVocabSize(config.numFeatures)
    val wordVectorizer = new CountVectorizer().setInputCol("words")
    val tagVectorizer = new CountVectorizer().setInputCol("tags")
    logger.info("Fitting the preprocessor pipeline. Please wait...")
    val preprocessor = new Pipeline().setStages(Array(labelIndexer, tokenizer, bofVectorizer, wordVectorizer, tagVectorizer))
      .fit(af.select("transition", "bof", "words", "tags"))
    val ef = preprocessor.transform(af)
    val ff = preprocessor.transform(bf)

    val labels = preprocessor.stages(0).asInstanceOf[StringIndexerModel].labels
    val labelSize = labels.length
    logger.info("#(labels) = " + labelSize)
    val bof = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
    logger.info("   #(bof) = " + bof.size)
    val vocabulary = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    logger.info(" #(words) = " + vocabulary.size)
    val vocabularyT = preprocessor.stages(4).asInstanceOf[CountVectorizerModel].vocabulary
    logger.info("  #(tags) = " + vocabularyT.size)

    // get the vocabulary and convert it to a map (word -> id), reserve an additional 0 for UNK word
    val vocab = vocabulary.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
    // get the vocabularyT and convert it to a map (tag -> id), reserve an additional 0 for UNK tag
    val vocabT = vocabularyT.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
    
    // create a udf to extract a seq of word ids and a sequence of tag ids from the stack and the queue, left-pad the seq with -1. 
    // set the maximum sequence of 10 (empirically found!)
    val maxSeqLen = 10
    val f = udf((stack: Seq[String], queue: Seq[String], words: Seq[String], tags: Seq[String]) => {
      val xs = queue.head +: stack
      val is = xs.map(x => vocab.getOrElse(words(x.toInt), 0))
      val js = xs.map(x => vocabT.getOrElse(tags(x.toInt), 0))
      val seqW = if (is.length < maxSeqLen) Seq.fill(maxSeqLen-is.length)(-1) ++ is else is.take(maxSeqLen)
      val seqT = if (js.length < maxSeqLen) Seq.fill(maxSeqLen-js.length)(-1) ++ js else js.take(maxSeqLen)
      seqW ++ seqT
    })

    val uf = ef.withColumn("seq", f(col("stack"), col("queue"), col("words"), col("tags"))).withColumn("y", g(col("label")))
    val vf = ff.withColumn("seq", f(col("stack"), col("queue"), col("words"), col("tags"))).withColumn("y", g(col("label")))

    uf.select("vec", "transition", "y", "weight", "seq").show(false)
    // prepare input data frame for multi-input BigDL model: featuresCol = Seq(vec, seq), labelCol=y
    import spark.implicits._
    import org.apache.spark.sql.types._
    
    val ufB = uf.select("vec", "seq", "y").map { row => 
      val x1 = row.getAs[Vector](0).toArray.map(_.toFloat)
      val x2 = row.getAs[Seq[Int]](1).toArray.map(_.toFloat)
      val y = row.getAs[Double](2).toFloat
      (x1 ++ x2, y)
    }.toDF("features", "y")
    val vfB = vf.select("vec", "seq", "y").map { row => 
      val x1 = row.getAs[Vector](0).toArray.map(_.toFloat)
      val x2 = row.getAs[Seq[Int]](1).toArray.map(_.toFloat)
      val y = row.getAs[Double](2).toFloat
      (x1 ++ x2, y)
    }.toDF("features", "y")

    ufB.show(5)
    ufB.printSchema
    
    // create a BigDL model
    val input1 = Input[Float](inputShape = Shape(bof.size))  // bof
    val input2 = Input[Float](inputShape = Shape(maxSeqLen)) // seqW
    val input3 = Input[Float](inputShape = Shape(maxSeqLen)) // seqT
    val dense1 = Dense[Float](config.featureEmbeddingSize).setName("dense1").inputs(input1)
    val maskingW = Masking[Float](-1).setName("maskingW").inputs(input2)
    val embeddingW = Embedding[Float](vocab.size + 1, config.wordEmbeddingSize, paddingValue = -1).setName("embeddingW").inputs(maskingW)
    val maskingT = Masking[Float](-1).setName("maskingT").inputs(input3)
    val embeddingT = Embedding[Float](vocabT.size + 1, config.tagEmbeddingSize, paddingValue = -1).setName("embeddingT").inputs(maskingT)
    // concat word and tag embeddings before passing to an LSTM
    val mergeWT = Merge.merge(inputs = List(embeddingW, embeddingT), mode = "concat")
    val lstm = LSTM[Float](config.recurrentSize).setName("lstm").inputs(mergeWT)
    // merge two branches
    val merge = Merge.merge(inputs = List(dense1, lstm), mode = "concat")
    val output = Dense[Float](labelSize, activation = "softmax").setName("output").inputs(merge)
    val model = Model[Float](Array(input1, input2, input3), output)
    model.summary()

    // overwrite the trained preprocessor
    val modelPath = Paths.get(config.modelPath, config.language, config.classifier).toString()
    preprocessor.write.overwrite().save(modelPath)
    // prepare a label weight tensor
    val weights = Tensor[Float](labelSize)
    val labelMap = uf.select("y", "weight").distinct.collect()
      .map(row => (row.getDouble(0).toInt, row.getDouble(1).toFloat)).toMap
    for (j <- labelMap.keys)
      weights(j) = labelMap(j)
    logger.info(labelMap.toString)

    val criterion = ClassNLLCriterion[Float](weights = weights, sizeAverage = false, logProbAsInput = false)
    val estimator = NNEstimator(model, criterion, Array(Array(bof.size), Array(maxSeqLen), Array(maxSeqLen)), Array(1))
    val trainingSummary = TrainSummary(appName = "rnnC", logDir = "sum/tdp/")
    val validationSummary = ValidationSummary(appName = "rnnC", logDir = "sum/tdp/")
    val numCores = Runtime.getRuntime.availableProcessors()
    val batchSize = numCores * 16
    estimator.setLabelCol("y").setFeaturesCol("features")
      .setBatchSize(batchSize)
      .setOptimMethod(new Adam(1E-4))
      .setMaxEpoch(config.iterations)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, vfB, Array(new Top1Accuracy[Float]()), batchSize)
    estimator.fit(ufB)
    model.saveModel(Paths.get(config.modelPath, config.language).toString() + "/rnnC.bigdl", overWrite = true)
  }

  /**
    * Trains a classifier to predict transition given a parsing config. This method uses an LSTM to extract features 
    * from the stack and the top token on the queue. It also uses the bof features for prediction as in the MLR or MLP model. 
    *
    * @param config 
    * @param graphs
    * @param devGraphs
    * @return a pipeline model
    */
  def trainB(config: ConfigTDP, graphs: List[Graph], devGraphs: List[Graph]) = {
    val af = addWeightCol(createDF(graphs))
    val bf = createDF(devGraphs) // no need to add weights for test df
    // bof preprocessing pipeline
    val labelIndexer = new StringIndexer().setInputCol("transition").setOutputCol("label").setHandleInvalid("skip")
    val tokenizer = new Tokenizer().setInputCol("bof").setOutputCol("tokens")
    val bofVectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("vec").setMinDF(config.minFrequency).setVocabSize(config.numFeatures)
    val wordVectorizer = new CountVectorizer().setInputCol("words")
    logger.info("Fitting the preprocessor pipeline. Please wait...")
    val preprocessor = new Pipeline().setStages(Array(labelIndexer, tokenizer, bofVectorizer, wordVectorizer))
      .fit(af.select("transition", "bof", "words"))
    val ef = preprocessor.transform(af)
    val ff = preprocessor.transform(bf)

    val labels = preprocessor.stages(0).asInstanceOf[StringIndexerModel].labels
    val labelSize = labels.length
    logger.info("#(labels) = " + labelSize)
    val bof = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
    logger.info("   #(bof) = " + bof.size)
    val vocabulary = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    logger.info(" #(words) = " + vocabulary.size)

    // get the vocabulary and convert it to a map (word -> id), reserve an additional 0 for UNK word
    val vocab = vocabulary.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
    
    // create a udf to extract a seq of words from the stack and the queue, left-pad the seq with -1. 
    // set the maximum sequence of 10 (empirically found!)
    val maxSeqLen = 10
    val f = udf((stack: Seq[String], queue: Seq[String], words: Seq[String]) => {
      val xs = queue.head +: stack
      val is = xs.map(x => vocab.getOrElse(words(x.toInt), 0))
      if (is.length < maxSeqLen) Seq.fill(maxSeqLen-is.length)(-1) ++ is else is.take(maxSeqLen)
    })

    // create seq features for LSTM
    val uf = ef.withColumn("seq", f(col("stack"), col("queue"), col("words"))).withColumn("y", g(col("label")))
    val vf = ff.withColumn("seq", f(col("stack"), col("queue"), col("words"))).withColumn("y", g(col("label")))

    uf.select("stack", "vec", "transition", "y", "weight", "seq").show(false)
    // prepare input data frame for multi-input BigDL model: featuresCol = Seq(vec, seq), labelCol=y
    import spark.implicits._
    import org.apache.spark.sql.types._
    
    val ufB = uf.select("vec", "seq", "y").map { row => 
      val x1 = row.getAs[Vector](0).toArray.map(_.toFloat)
      val x2 = row.getAs[Seq[Int]](1).toArray.map(_.toFloat)
      val y = row.getAs[Double](2).toFloat
      (x1 ++ x2, y)
    }.toDF("features", "y")
    val vfB = vf.select("vec", "seq", "y").map { row => 
      val x1 = row.getAs[Vector](0).toArray.map(_.toFloat)
      val x2 = row.getAs[Seq[Int]](1).toArray.map(_.toFloat)
      val y = row.getAs[Double](2).toFloat
      (x1 ++ x2, y)
    }.toDF("features", "y")

    ufB.show(5)
    ufB.printSchema
    
    val input1 = Input[Float](inputShape = Shape(bof.size))
    val input2 = Input[Float](inputShape = Shape(maxSeqLen))
    val dense1 = Dense[Float](config.featureEmbeddingSize).setName("dense1").inputs(input1)
    val masking = Masking[Float](-1).setName("masking").inputs(input2)
    val embedding = Embedding[Float](vocab.size + 1, config.wordEmbeddingSize, paddingValue = -1).setName("embedding").inputs(masking)
    val lstm = LSTM[Float](config.recurrentSize).setName("lstm").inputs(embedding)
    // merge two branches
    val merge = Merge.merge(inputs = List(dense1, lstm), mode = "concat")
    val output = Dense[Float](labelSize, activation = "softmax").setName("output").inputs(merge)
    val model = Model[Float](Array(input1, input2), output)
    model.summary()

    // overwrite the trained preprocessor
    val modelPath = Paths.get(config.modelPath, config.language, config.classifier).toString()
    preprocessor.write.overwrite().save(modelPath)
    // prepare a label weight tensor
    val weights = Tensor[Float](labelSize)
    val labelMap = uf.select("y", "weight").distinct.collect()
      .map(row => (row.getDouble(0).toInt, row.getDouble(1).toFloat)).toMap
    for (j <- labelMap.keys)
      weights(j) = labelMap(j)
    logger.info(labelMap.toString)

    val criterion = ClassNLLCriterion[Float](weights = weights, sizeAverage = false, logProbAsInput = false)
    val estimator = NNEstimator(model, criterion, Array(Array(bof.size), Array(maxSeqLen)), Array(1))
    val trainingSummary = TrainSummary(appName = "rnnB", logDir = "sum/tdp/")
    val validationSummary = ValidationSummary(appName = "rnnB", logDir = "sum/tdp/")
    val numCores = Runtime.getRuntime.availableProcessors()
    val batchSize = numCores * 16
    estimator.setLabelCol("y").setFeaturesCol("features")
      .setBatchSize(batchSize)
      .setOptimMethod(new Adam(1E-4))
      .setMaxEpoch(config.iterations)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, vfB, Array(new Top1Accuracy[Float]()), batchSize)
    estimator.fit(ufB)
    model.saveModel(Paths.get(config.modelPath, config.language).toString() + "/rnnB.bigdl", overWrite = true)
  }

  /**
    * Trains a classifier to predict transition given a parsing config. This method uses an LSTM to extract features 
    * from the stack and the top token on the queue. 
    *
    * @param config 
    * @param graphs
    * @param devGraphs
    * @return a pipeline model
    */
  def trainA(config: ConfigTDP, graphs: List[Graph], devGraphs: List[Graph]) = {
    val af = addWeightCol(createDF(graphs))
    val bf = createDF(devGraphs)
    val labelIndexer = new StringIndexer().setInputCol("transition").setOutputCol("label").setHandleInvalid("skip")
    val wordVectorizer = new CountVectorizer().setInputCol("words")
    logger.info("Fitting the preprocessor pipeline. Please wait...")
    val preprocessor = new Pipeline().setStages(Array(labelIndexer, wordVectorizer)).fit(af.select("transition", "words"))
    val ef = preprocessor.transform(af)
    val ff = preprocessor.transform(bf)

    val labels = preprocessor.stages(0).asInstanceOf[StringIndexerModel].labels
    val labelSize = labels.length
    logger.info("  #(labels) = " + labelSize)    
    // get the vocabulary and convert it to a map (word -> id), reserve 0 for UNK token
    val vocab = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
    logger.info(s"#(words) = ${vocab.size}")
    
    // create a udf to extract a seq of tokens from the stack and the queue, left-pad the seq with -1. 
    // set the maximum sequence of 10 (empirically found!)
    val maxSeqLen = 10
    val f = udf((stack: Seq[String], queue: Seq[String], words: Seq[String]) => {
      val xs = queue.head +: stack
      val is = xs.map(x => vocab.getOrElse(words(x.toInt), 0))
      if (is.length < maxSeqLen) Seq.fill(maxSeqLen-is.length)(-1) ++ is else is.take(maxSeqLen)
    })

    val uf = ef.withColumn("seq", f(col("stack"), col("queue"), col("words"))).withColumn("y", g(col("label")))
    val vf = ff.withColumn("seq", f(col("stack"), col("queue"), col("words"))).withColumn("y", g(col("label")))
    uf.select("transition", "y", "weight", "seq").show(false)

    val sequential = Sequential[Float]()
    val masking = Masking[Float](-1, inputShape = Shape(maxSeqLen)).setName("masking")
    sequential.add(masking)
    val embedding = Embedding[Float](vocab.size + 1, config.wordEmbeddingSize, paddingValue = -1).setName("embedding")
    sequential.add(embedding)
    val lstm = LSTM[Float](config.recurrentSize).setName("lstm")
    sequential.add(lstm)
    val output = Dense[Float](labelSize, activation = "softmax").setName("output")
    sequential.add(output)
    sequential.summary()

    // overwrite the trained preprocessor
    val modelPath = Paths.get(config.modelPath, config.language, config.classifier).toString()
    preprocessor.write.overwrite().save(modelPath)
    // prepare a label weight tensor
    val weights = Tensor[Float](labelSize)
    val labelMap = uf.select("y", "weight").distinct.collect()
      .map(row => (row.getDouble(0).toInt, row.getDouble(1).toFloat)).toMap
    for (j <- labelMap.keys)
      weights(j) = labelMap(j)
    logger.info(labelMap.toString)

    val criterion = ClassNLLCriterion[Float](weights = weights, sizeAverage = false, logProbAsInput = false)
    val estimator = NNEstimator(sequential, criterion, Array(maxSeqLen), Array(1))
    val trainingSummary = TrainSummary(appName = "rnnA", logDir = "sum/tdp/")
    val validationSummary = ValidationSummary(appName = "rnnA", logDir = "sum/tdp/")
    val numCores = Runtime.getRuntime.availableProcessors()
    val batchSize = numCores * 16
    estimator.setLabelCol("y").setFeaturesCol("seq")
      .setBatchSize(batchSize)
      .setOptimMethod(new Adam(1E-4))
      .setMaxEpoch(config.iterations)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, vf, Array(new Top1Accuracy[Float]()), batchSize)
    estimator.fit(uf)
    sequential.saveModel(Paths.get(config.modelPath, config.language).toString() + "/rnnA.bigdl", overWrite = true)
  }

  /**
    * Evaluates the accuracy of the transition classifier, including overall accuracy,
    * precision, recall and f-measure ratios for each transition label.
    *
    * @param modelPath trained model path
    * @param graphs    a list of dependency graphs
    * @param discrete                 
    */
  def eval(modelPath: String, graphs: List[Graph], discrete: Boolean = true): Unit = {
    val model = PipelineModel.load(modelPath)
    val inputDF = createDF(graphs)
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
      opt[String]('c', "classifier").action((x, conf) => conf.copy(classifier = x)).text("classifier")
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
            classifier.eval(modelPath, trainingGraphs)
            // classifier.evalManual(modelPath, developmentGraphs)
            // classifier.evalManual(modelPath, trainingGraphs)
          }
          case "train" => {
            config.classifier match {
              case "rnnC" => {
                Engine.init
                classifier.trainC(config, trainingGraphs, developmentGraphs)
              }
              case "rnnB" => {
                Engine.init
                classifier.trainB(config, trainingGraphs, developmentGraphs)
              }
              case "rnnA" => {
                Engine.init
                classifier.trainA(config, trainingGraphs, developmentGraphs)
              }
              case _ =>
                val hiddenLayers = if (config.hiddenUnits.isEmpty) Array[Int](); else config.hiddenUnits.split("[,\\s]+").map(_.toInt)
                classifier.train(modelPath, trainingGraphs, hiddenLayers)
            }
          }
          case "test" => 
          case "precompute" =>
            // load a pretrained transition models and compute vectors of all past transition sequences 
            val modelPath = "bin/asp/eng.bigdl"
            val model = Try(Models.loadModel[Float](modelPath)).getOrElse {
              logger.info("First load failed, retrying...")
              Models.loadModel[Float](modelPath)
            }
            model.summary()

            // prepare contexts
            val (uf, vf) = (
              classifier.createDF(trainingGraphs).select("pastTransitions"), 
              classifier.createDF(developmentGraphs).select("pastTransitions")
            )
            import org.apache.spark.sql.functions._
            val uvf = uf.union(vf).filter(size(col("pastTransitions")) > 0)
            logger.info("#(total contexts) = " + uvf.count())

            uvf.show(40, false)
            // compute pretrained vectors
            val pretrainerConfig = PretrainerConfig()
            import spark.implicits._
            val output = uvf.map { row => 
              val xs = row.getAs[Seq[String]](0)
              val ys = TransitionPretrainer.compute(model, pretrainerConfig, xs)
              (xs, ys)
            }.toDF("xs", "ys")
            output.show(20)
            output.write.format("json").save(s"dat/bin/asp/${config.language}/pretrained")
        }
        spark.stop()
      case None =>
    }
  }
}

