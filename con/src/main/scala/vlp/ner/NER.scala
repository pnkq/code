package vlp.ner

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.embeddings.{BertEmbeddings, DeBertaEmbeddings, DistilBertEmbeddings}
import com.johnsnowlabs.nlp.{DocumentAssembler, EmbeddingsFinisher}
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
import com.johnsnowlabs.nlp.training.CoNLL
import scala.io.Source

import org.json4s._
import org.json4s.jackson.Serialization
import scopt.OptionParser

import java.nio.file.{Files, Paths, StandardOpenOption}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.ml.linalg.DenseVector

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.models.{Models, KerasNet}
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.dllib.nnframes.{NNModel, NNEstimator}
import com.intel.analytics.bigdl.dllib.nn.{TimeDistributedCriterion, ClassNLLCriterion}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.optim.Trigger


case class ScoreNER(
  modelType: String,
  split: String,
  accuracy: Double,
  confusionMatrix: Matrix,
  precision: Array[Double],
  recall: Array[Double],
  fMeasure: Array[Double]
)

/**
  * phuonglh, April 2023
  * 
  * An implementation of Vietnamese NER on a medical data set using pretrained models. 
  * 
  */

object NER {
  implicit val formats: AnyRef with Formats = Serialization.formats(NoTypeHints)
  private val labelIndex = Map[String, Int](
    "O" -> 1, "B-problem" -> 2, "I-problem" -> 3, "B-treatment" -> 4, "I-treatment" -> 5, "B-test" -> 6, "I-test" -> 7
  )
  val labelDict: Map[Double, String] = labelIndex.keys.map(k => (labelIndex(k).toDouble, k)).toMap
  private val numCores = Runtime.getRuntime.availableProcessors()

  /**
    * Builds a pipeline for BigDL model: sequencer -> flattener -> padder
    *
    * @param config config
    */
  private def pipelineBigDL(config: ConfigNER): Pipeline = {
    // use a label sequencer to transform `ys` into sequences of integers (1-based, for BigDL to work)
    // we use padding value -1f for padding label sequences.
    val sequencer = new Sequencer(labelIndex, config.maxSeqLen, -1f).setInputCol("ys").setOutputCol("target")
    val flattener = new FeatureFlattener().setInputCol("xs").setOutputCol("as")
    val padder = new FeaturePadder(config.maxSeqLen*768, 0f).setInputCol("as").setOutputCol("features")
    new Pipeline().setStages(Array(sequencer, flattener, padder))
  }

  private def prepareSnowPreprocessor(config: ConfigNER, trainingDF: DataFrame): PipelineModel = {
    val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val tokenizer = new Tokenizer().setInputCols(Array("document")).setOutputCol("token")
    val embeddings = config.modelType match {
      case "b" => BertEmbeddings.pretrained("bert_base_multilingual_cased", "xx").setInputCols("document", "token").setOutputCol("embeddings").setCaseSensitive(true)
      case "d" => DeBertaEmbeddings.pretrained("deberta_embeddings_vie_small", "vie").setInputCols("document", "token").setOutputCol("embeddings").setCaseSensitive(true)
      case "m" => DeBertaEmbeddings.pretrained("mdeberta_v3_base", "xx").setInputCols("document", "token").setOutputCol("embeddings").setCaseSensitive(true)
      case "s" => DistilBertEmbeddings.pretrained("distilbert_base_cased", "vi").setInputCols("document", "token").setOutputCol("embeddings").setCaseSensitive(true)
      case "x" => XlmRoBertaEmbeddings.pretrained("xlm_roberta_large", "xx").setInputCols("document", "token").setOutputCol("embeddings").setCaseSensitive(true) // _large / _base
      case _ => DeBertaEmbeddings.pretrained("deberta_embeddings_vie_small", "vie").setInputCols("document", "token").setOutputCol("embeddings").setCaseSensitive(true)
    }
    val finisher = new EmbeddingsFinisher().setInputCols("embeddings").setOutputCols("xs").setOutputAsVector(false) // output as arrays
    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings, finisher))
    val preprocessor = pipeline.fit(trainingDF)
    preprocessor.write.overwrite.save(config.modelPath + "/" + config.modelType)
    preprocessor
  }

  private def labelWeights(spark: SparkSession, df: DataFrame, labelCol: String): Tensor[Float] = {
    import spark.implicits._
    // select non-padded labels
    val ef = df.select(labelCol).flatMap(row => row.getAs[DenseVector](0).toArray.filter(_ > 0))
    val ff = ef.groupBy("value").count // two columns [value, count]
    // count their frequencies
    val total: Long = ff.agg(sum("count")).head.getLong(0)
    val numLabels: Long = ff.count()
    val wf = ff.withColumn("w", lit(total.toDouble/numLabels)/col("count")).sort("value") // sort to align label indices
    val w = wf.select("w").collect().map(row => row.getDouble(0)).map(_.toFloat)
    Tensor[Float](w, Array(w.length))
  }


  /**
    * Trains a NER model using the BigDL framework with user-defined model. This approach is more flexible than the [[trainJSL()]] method.
    * @param config a config
    * @param trainingDF a training df
    * @param developmentDF a development df
   * @param firstTime run the first time, which will create and save parquet preprocessed datasets.
    * @return a preprocessor and a BigDL model
    */
  private def trainBDL(config: ConfigNER, trainingDF: DataFrame, developmentDF: DataFrame, firstTime: Boolean = false): (PipelineModel, KerasNet[Float]) = {
    val (preprocessorSnow, af, bf) = if (firstTime) {
      val preprocessorSnow = prepareSnowPreprocessor(config, trainingDF)
      println("Applying the Snow preprocessor to (training, dev.) datasets...")
      val (af, bf) = (preprocessorSnow.transform(trainingDF), preprocessorSnow.transform(developmentDF))
      af.write.mode("overwrite").parquet(config.modelPath + "/af")
      af.write.mode("overwrite").parquet(config.modelPath + "/bf")
      (preprocessorSnow, af, bf)
    } else {
      val preprocessorSnow = PipelineModel.load(config.modelPath + "/" + config.modelType)
      val spark = SparkSession.getActiveSession.get
      val af = spark.read.parquet(config.modelPath + "/af")
      val bf = spark.read.parquet(config.modelPath + "/bf")
      (preprocessorSnow, af, bf)
    }
    // supplement pipeline for BigDL
    val preprocessorBigDL = pipelineBigDL(config).fit(af)
    val (uf, vf) = (preprocessorBigDL.transform(af), preprocessorBigDL.transform(bf))
    
    // create a BigDL model
    val bigdl = Sequential()
    bigdl.add(Reshape(targetShape=Array(config.maxSeqLen, 768), inputShape=Shape(config.maxSeqLen*768)).setName("reshape"))
    for (j <- 1 to config.layers) {
      bigdl.add(Bidirectional(LSTM(outputDim = config.hiddenSize, returnSequences = true).setName(s"LSTM-$j")))
    }
    bigdl.add(Dropout(0.1).setName("dropout"))
    bigdl.add(Dense(labelIndex.size, activation="softmax").setName("dense"))
    val (featureSize, labelSize) = (Array(config.maxSeqLen*768), Array(config.maxSeqLen))
    val w = labelWeights(SparkSession.getActiveSession.get, uf, "target")
    // should set the sizeAverage=false in ClassNLLCriterion
    val criterion = ClassNLLCriterion(weights = w, sizeAverage = false, paddingValue = -1)
    val estimator = NNEstimator(bigdl, TimeDistributedCriterion(criterion, sizeAverage = true), featureSize, labelSize)
    val trainingSummary = TrainSummary(appName = config.modelType, logDir = "sum/med/")
    val validationSummary = ValidationSummary(appName = config.modelType, logDir = "sum/med/")
    val batchSize = if (config.batchSize % numCores != 0) numCores * 4; else config.batchSize
    estimator.setLabelCol("target").setFeaturesCol("features")
      .setBatchSize(batchSize)
      .setOptimMethod(new Adam(config.learningRate))
      .setMaxEpoch(config.epochs)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, vf, Array(new TimeDistributedTop1Accuracy(paddingValue = -1)), batchSize)
    estimator.fit(uf)
    (preprocessorSnow, bigdl)
  }


  /**
    * Trains a NER model using the JohnSnowLab [[NerDLApproach]]. This is a CNN-BiLSTM-CRF network model, which is readily usable but not 
    * flexible enough compared to our own NER model using BigDL.
    *
    * @param config a config
    * @param trainingDF a df
    * @param developmentDF a df
    */
  private def trainJSL(config: ConfigNER, trainingDF: DataFrame, developmentDF: DataFrame): PipelineModel = {
    val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val tokenizer = new Tokenizer().setInputCols(Array("document")).setOutputCol("token")
    val embeddings = config.modelType match {
      case "m" => DeBertaEmbeddings.pretrained("mdeberta_v3_base", "xx").setInputCols("document", "token").setOutputCol("embeddings")
      case "d" => DeBertaEmbeddings.pretrained("deberta_embeddings_vie_small", "vie").setInputCols("document", "token").setOutputCol("embeddings").setCaseSensitive(true)
      case "s" => DistilBertEmbeddings.pretrained("distilbert_base_cased", "vi").setInputCols("document", "token").setOutputCol("embeddings")
      case _ => DeBertaEmbeddings.pretrained("deberta_embeddings_vie_small", "vie").setInputCols("document", "token").setOutputCol("embeddings").setCaseSensitive(true)
    }
    // val finisher = new EmbeddingsFinisher().setInputCols("embeddings").setOutputCols("xs").setOutputAsVector(true).setCleanAnnotations(false)
    val stages = Array(document, tokenizer, embeddings)
    // train a preprocessor 
    val preprocessor = new Pipeline().setStages(stages)
    val preprocessorModel = preprocessor.fit(trainingDF)
    // use the preprocessor pipeline to transform the data sets
    val df = preprocessorModel.transform(developmentDF)
    df.write.mode("overwrite").parquet(config.validPath)
    val tagger = new NerDLApproach().setInputCols(Array("document", "token", "embeddings"))
      .setLabelColumn("label").setOutputCol("ner")
      .setMaxEpochs(config.epochs)
      .setLr(config.learningRate.toFloat).setPo(0.005f)
      .setBatchSize(config.batchSize).setRandomSeed(0)
      .setVerbose(0)
      .setValidationSplit(0.2f)
      // .setEvaluationLogExtended(false).setEnableOutputLogs(false).setIncludeConfidence(true)
      .setEnableMemoryOptimizer(true)
      .setTestDataset(config.validPath)
    val pipeline = new Pipeline().setStages(stages ++ Array(tagger))
    val model = pipeline.fit(trainingDF)
    model
  }

  def evaluate(result: DataFrame, config: ConfigNER, split: String): ScoreNER = {
    val predictionsAndLabels = result.rdd.map { row =>
      // prediction
      val zs = row.getAs[Seq[Float]](0).toArray.map(_.toDouble)
      // label
      val ys = row.getAs[DenseVector](1).toArray
      // remove padding values -1 of the target from the evaluation
      val pairs = zs.zip(ys).filter(_._2 != -1d)
      (pairs.map(_._1), pairs.map(_._2))
    }.flatMap { case (prediction, label) => prediction.zip(label) }
    val metrics = new MulticlassMetrics(predictionsAndLabels)
    val ls = metrics.labels
    val numLabels = ls.max.toInt + 1 // zero-based labels
    val precisionByLabel = Array.fill(numLabels)(0d)
    val recallByLabel = Array.fill(numLabels)(0d)
    val fMeasureByLabel = Array.fill(numLabels)(0d)
    ls.foreach { k => 
      precisionByLabel(k.toInt) = metrics.precision(k)
      recallByLabel(k.toInt) = metrics.recall(k)
      fMeasureByLabel(k.toInt) = metrics.fMeasure(k)
    }
    ScoreNER(
      config.modelType, split,
      metrics.accuracy, metrics.confusionMatrix, 
      precisionByLabel, recallByLabel, fMeasureByLabel
    )
  }

  private def saveScore(score: ScoreNER, path: String) = {
    val content = Serialization.writePretty(score) + ",\n"
    Files.write(Paths.get(path), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
  }

  /**
    * Exports result data frame (2-col format) into a text file of CoNLL-2003 format for 
    * evaluation with CoNLL evaluation script (correct <space> prediction).
    * @param result a data frame of two columns "prediction, target"
    * @param config a config
    * @param split a split name
    */
  private def export(result: DataFrame, config: ConfigNER, split: String) = {
    val spark = SparkSession.getActiveSession.get
    import spark.implicits._
    val ss = result.map { row => 
      val prediction = row.getSeq[String](0)
      val target = row.getSeq[String](1)
      val lines = target.zip(prediction).map(p => p._1 + " " + p._2)
      lines.mkString("\n") + "\n"
    }.collect()
    val s = ss.mkString("\n")
    Files.write(Paths.get(s"${config.outputPath}/${config.modelType}-$split.txt"), s.getBytes, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }

  def predict(bigdl: KerasNet[Float], bf: DataFrame, config: ConfigNER, argmax: Boolean=true): DataFrame = {
    val preprocessorBigDL = pipelineBigDL(config).fit(bf)
    val vf = preprocessorBigDL.transform(bf)
    // convert bigdl to sequential model
    val sequential = bigdl.asInstanceOf[Sequential[Float]]
    // bigdl produces 3-d output results (including batch dimension), we need to convert it to 2-d results.
    if (argmax)
      sequential.add(ArgMaxLayer())
    sequential.summary()
    // wrap to a Spark model and run prediction
    val model = NNModel(sequential)
    model.transform(vf)
  }

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[ConfigNER](getClass.getName) {
      head(getClass.getName, "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 16g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either {eval, train, predict}")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('j', "layers").action((x, conf) => conf.copy(layers = x)).text("number of RNN layers or Transformer blocks")
      opt[Int]('h', "hiddenSize").action((x, conf) => conf.copy(hiddenSize = x)).text("encoder hidden size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x)).text("learning rate, default value is 1E-5")
      opt[String]('d', "trainPath").action((x, conf) => conf.copy(trainPath = x)).text("training data directory")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model folder, default is 'bin/'")
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path")
      opt[String]('s', "scorePath").action((x, conf) => conf.copy(scorePath = x)).text("score path")
      opt[Boolean]('f', "firstTime").action((_, conf) => conf.copy(firstTime = true)).text("first time in the training")
    }
    opts.parse(args, ConfigNER()) match {
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

        // read the df using the CoNLL format of Spark-NLP, which provides some columns, including [text, label] columns.
        val source = Source.fromFile(config.trainPath, "UTF-8")
        val df = CoNLL(conllLabelIndex = 3).readDatasetFromLines(source.getLines.toArray, spark).toDF
        val af = df.withColumn("ys", col("label.result")).withColumn("length", size(col("ys")))
        println(s"Number of samples = ${af.count}")
        // remove short sentences
        val ef = af.filter(col("length") >= config.minSeqLen)
        println(s"Number of samples >= ${config.minSeqLen} = ${ef.count}")
        // split train/dev.
        val Array(trainingDF, developmentDF) = ef.randomSplit(Array(0.8, 0.2), 220712L)
        developmentDF.show()
        developmentDF.printSchema()
        developmentDF.select("token.result", "ys").show(3, truncate = false)

        val modelPath = config.modelPath + "/" + config.modelType
        config.mode match {
          case "train" =>
            // val model = trainJSL(config, trainingDF, developmentDF)
            val (_, bigdl) = trainBDL(config, trainingDF, developmentDF, config.firstTime)
            bigdl.saveModel(modelPath + "/ner.bigdl", overWrite = true)
            val bf = spark.read.parquet(config.modelPath + "/bf")
            val output = predict(bigdl, bf, config)
            output.show
          case "predict" =>
          case "evalBDL" => 
            val bigdl = Models.loadModel[Float](modelPath + "/ner.bigdl")
            // load preprocessed training/dev. data frames
            val af = spark.read.parquet(config.modelPath + "/af")
            println(s"Number of training samples = ${af.count}")
            // training result
            val outputTrain = predict(bigdl, af, config)
            outputTrain.show
            outputTrain.printSchema
            val trainResult = outputTrain.select("prediction", "target")
            var score = evaluate(trainResult, config, "train")
            saveScore(score, config.scorePath)
            // validation result
            val bf = spark.read.parquet(config.modelPath + "/bf")
            println(s"Number of validation samples = ${bf.count}")
            val outputValid = predict(bigdl, bf, config, argmax = false)
            outputValid.show
            val validResult = outputValid.select("prediction", "target")
            score = evaluate(validResult, config, "valid")
            saveScore(score, config.scorePath)
            // convert "prediction" column to human-readable label column "zs"
            val sequencerPrediction = new SequencerDouble(labelDict).setInputCol("prediction").setOutputCol("zs")
            val uf = sequencerPrediction.transform(outputTrain)
            val vf = sequencerPrediction.transform(outputValid)
            // export to CoNLL format
            export(uf.select("zs", "ys"), config, "train")
            export(vf.select("zs", "ys"), config, "valid")
          case "evalJSL" => 
            val model = PipelineModel.load(modelPath)
            val tf = model.transform(trainingDF).withColumn("zs", col("ner.result")).withColumn("ys", col("label.result"))
            val sequencerPrediction = new SequencerNER(labelIndex).setInputCol("zs").setOutputCol("prediction")
            val sequencerTarget = new SequencerNER(labelIndex).setInputCol("ys").setOutputCol("target")
            // training result
            val af = sequencerTarget.transform(sequencerPrediction.transform(tf))
            val trainResult = af.select("prediction", "target")
            var score = evaluate(trainResult, config, "train")
            saveScore(score, config.scorePath)
            // validation result            
            val vf = model.transform(developmentDF).withColumn("zs", col("ner.result")).withColumn("ys", col("label.result"))
            val bf = sequencerTarget.transform(sequencerPrediction.transform(vf))
            val validResult = bf.select("prediction", "target")
            score = evaluate(validResult, config, "valid")
            saveScore(score, config.scorePath)
            validResult.show(5, truncate = false)
            // export to CoNLL format
            export(af.select("zs", "ys"), config, "train")
            export(bf.select("zs", "ys"), config, "valid")
          case _ => println("Invalid running mode!")
        }

        sc.stop()
      case None =>
    }

  }
}
