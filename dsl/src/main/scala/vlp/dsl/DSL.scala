package vlp.dsl

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, RegexTokenizer, StringIndexer, StringIndexerModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import scopt.OptionParser

import java.nio.file.{Files, Paths, StandardOpenOption}
import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.keras.models.{KerasNet, Models}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.{Loss, Top1Accuracy, Trigger}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach
import com.johnsnowlabs.nlp.embeddings.XlmRoBertaSentenceEmbeddings
import org.apache.spark.SparkConf
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, lit, monotonically_increasing_id, udf}

import java.nio.file.{Files, Paths}
import scala.io.Source


/**
 * Implementation of an approach for the DSL-ML-20234 shared task at VarDial-2024.
 * <p/>
 * phuonglh@gmail.com
 */
object DSL {

  private def readPath(spark: SparkSession, path: String): DataFrame = {
    spark.read.option("delimiter", "\t")
      .csv(path).toDF("languages", "text")
      .withColumn("id", monotonically_increasing_id)
  }

  private def preprocess(df: DataFrame, vocabPath: String): PipelineModel = {
    val indexerLang = new StringIndexer().setInputCol("languages").setOutputCol("label").setHandleInvalid("skip")
    val tokenizerText = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern("""[\s.,"?!“”)(’'`:/«*»@\]\[–—-]+""")
    val vectorizerToken = new CountVectorizer().setInputCol("tokens").setOutputCol("features").setMinDF(2)
    val pipeline = new Pipeline().setStages(Array(indexerLang, tokenizerText, vectorizerToken))
    val model = pipeline.fit(df)
    // save the vocab
    val vocabulary = model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
    Files.write(Paths.get(vocabPath), vocabulary.mkString("\n").getBytes, StandardOpenOption.CREATE)
    model
  }

  private def jslTrain(config: Config, df: DataFrame): PipelineModel = {
    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val tokenEmbedding = XlmRoBertaSentenceEmbeddings.pretrained("sent_xlm_roberta_base", "xx")
      .setInputCols("document").setOutputCol("embeddings")
    val validPath = s"dat/dsl/${config.language}"
    if (!Files.exists(Paths.get(validPath))) {
      val preprocessor = new Pipeline().setStages(Array(documentAssembler, tokenEmbedding))
      val preprocessorModel = preprocessor.fit(df)
      val ef = preprocessorModel.transform(df)
      ef.write.parquet(validPath)
    }
    val classifier = new ClassifierDLApproach().setInputCols("embeddings").setOutputCol("prediction").setLabelColumn("languages")
      .setBatchSize(config.batchSize).setMaxEpochs(config.epochs).setLr(5E-5f)
      .setTestDataset(validPath)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenEmbedding, classifier))
    println("Training the JSL pipeline...")
    val model = pipeline.fit(df)
    model.write.overwrite().save(s"bin/jsl/${config.language}")
    model
  }

  private def jslTest(spark: SparkSession, config: Config, pipeline: Option[PipelineModel]) = {
    val df = if (config.testMode) {
      val testPath = s"dat/DSL-ML-2024/${config.language}/${config.language}_test.tsv"
      val bf = spark.read.option("delimiter", "\t").csv(testPath).toDF("text")
      // add a fake "languages" column for the preprocessor pipeline to work; this column is not used in test mode anyway
      val fakeLang = config.language match {
        case "EN" => "EN-US"
        case "ES" => "ES-ES"
        case "FR" => "FR-FR"
        case "PT" => "PT-PT"
        case "BCMS" => "sr"
      }
      bf.withColumn("languages", lit(fakeLang))
    } else {
      val testPath = s"dat/DSL-ML-2024/${config.language}/${config.language}_dev.tsv"
      spark.read.option("delimiter", "\t").csv(testPath).toDF("languages", "text")
    }
    val model = pipeline match {
      case Some(model) => model
      case None => PipelineModel.load(s"bin/jsl/${config.language}")
    }
    val ef = model.transform(df)
    import spark.implicits._
    ef.select("prediction").show(false)
    ef.select("prediction.result").show(false)
    val output = ef.select("prediction.result").map { row => row.getSeq[String](0).head }.collect()
    Files.write(Paths.get(s"out/${config.language}-open-VLP-run-3"), output.mkString("\n").getBytes(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }

  private def eval(config: Config, uf: DataFrame, vf: DataFrame, bigdl: Option[Sequential[Float]]) = {
    val model = bigdl match {
      case Some(network) => network
      case None =>
        var modelPath = s"${config.modelPath}/rnn/${config.language}-${config.modelType}"
        if (config.pretrained) modelPath += "-x"
        println(s"Loading model in the path: $modelPath...")
        Models.loadModel(modelPath)
    }
    val ufZ = model.predict(uf, Array("x"), "z")
    val vfZ = model.predict(vf, Array("x"), "z")
    (ufZ.select("label", "z"), vfZ.select("label", "z"))
  }

  /**
   * Convert input data frames into BigDL-ready (1-based label index vector).
   * @param config
   * @param tokenMap
   * @param train
   * @param valid
   * @return training and dev. data frames
   */
  def bigdlPreprocess(config: Config, tokenMap: Map[String, Int], train: DataFrame, valid: DataFrame) = {
    // create a sequencer to convert a sequence of tokens into a sequence of indices
    val tokenSequencer = new Sequencer(tokenMap, config.maxSeqLen, 0f).setInputCol("tokens").setOutputCol("x")
    val (uf1, vf1) = (tokenSequencer.transform(train), tokenSequencer.transform(valid))
    // increase the label to 1 for BigDL to work
    (
      uf1.withColumn("y", col("label") + lit(1)),
      vf1.withColumn("y", col("label") + lit(1)),
    )
  }

  // user-defined function for argmax
  private def argmax: UserDefinedFunction = udf((x: Seq[Float]) => {
    x.zipWithIndex.maxBy(_._1)._2.toDouble
  })

  private def predict(config: Config, tokenMap: Map[String, Int], labels: Array[String], spark: SparkSession,
              preprocessor: PipelineModel, bigdl: Option[KerasNet[Float]]) = {
    val model = bigdl match {
      case Some(network) => network
      case None =>
        var modelPath = s"${config.modelPath}/rnn/${config.language}-${config.modelType}"
        if (config.pretrained) modelPath += "-x"
        println(s"Loading model in the path: $modelPath...")
        Models.loadModel(modelPath)
    }
    val df = if (config.testMode) {
      val testPath = s"dat/DSL-ML-2024/${config.language}/${config.language}_test.tsv"
      val bf = spark.read.option("delimiter", "\t").csv(testPath).toDF("text").withColumn("id", monotonically_increasing_id)
      // add a fake "languages" column for the preprocessor pipeline to work; this column is not used in test mode anyway
      val fakeLang = config.language match {
        case "EN" => "EN-US"
        case "ES" => "ES-ES"
        case "FR" => "FR-FR"
        case "PT" => "PT-PT"
        case "BCMS" => "sr"
      }
      bf.withColumn("languages", lit(fakeLang))
    } else {
      val testPath = s"dat/DSL-ML-2024/${config.language}/${config.language}_dev.tsv"
      spark.read.option("delimiter", "\t").csv(testPath).toDF("languages", "text").withColumn("id", monotonically_increasing_id)
    }
    // run the test df through the preprocessor
    val ef = preprocessor.transform(df)
    // create a sequencer to convert a sequence of tokens into a sequence of indices
    val tokenSequencer = new Sequencer(tokenMap, config.maxSeqLen, 0f).setInputCol("tokens").setOutputCol("x")
    val ff = tokenSequencer.transform(ef)
    // run the test ff through the network
    val uf = model.predict(ff, Array("x"), "z")
    val zf = uf.withColumn("prediction", argmax(col("z")))
    zf.select("prediction", "tokens").show()
    // collect the prediction label and write out the result
    import spark.implicits._
    val prediction = zf.select("prediction").map { row =>
      val j = row.getDouble(0).toInt
      labels(j)
    }.collect()
    // need to treat skipped line in prediction (on the development set only)...
    val run = if (config.pretrained) 2 else 1
    val outputPath = s"out/${config.language}-closed-VLP-run-$run.txt"
    Files.write(Paths.get(outputPath), prediction.mkString("\n").getBytes, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[Config](getClass.getName) {
      head(getClass.getName, "1.0")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 16g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either {eval, train, predict}")
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('j', "layers").action((x, conf) => conf.copy(layers = x)).text("number of RNN layers or Transformer blocks")
      opt[Int]('w', "tokenEmbeddingSize").action((x, conf) => conf.copy(tokenEmbeddingSize = x)).text("token embedding size")
      opt[Int]('h', "tokenHiddenSize").action((x, conf) => conf.copy(tokenHiddenSize = x)).text("encoder hidden size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model folder, default is 'bin/'")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path")
      opt[String]('s', "scorePath").action((x, conf) => conf.copy(scorePath = x)).text("score path")
      opt[Unit]('x', "pretrained").action((_, conf) => conf.copy(pretrained = true)).text("use pretrained flag")
      opt[Unit]('y', "testMode").action((_, conf) => conf.copy(testMode = true)).text("test mode for preparing submission")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        val conf = new SparkConf().setAppName(getClass.getName).setMaster("local[*]")
          .set("spark.driver.memory", config.driverMemory)
        // Creates or gets SparkContext with optimized configuration for BigDL performance.
        // The method will also initialize the BigDL engine.
        val sc = NNContext.initNNContext(conf)
        sc.setLogLevel("ERROR")
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()

        val languages = Seq("EN", "ES", "FR", "PT", "BCMS")
        val basePath = "dat/DSL-ML-2024"
        val paths = languages.map { language =>
          (s"$basePath/$language/${language}_train.tsv", s"$basePath/$language/${language}_dev.tsv")
        }
        val languageIndex = languages.zipWithIndex.toMap
        val (dfT0, dfV) = (
          readPath(spark, paths(languageIndex(config.language))._1),
          readPath(spark, paths(languageIndex(config.language))._2)
        )
        // trainingDF, if in test mode, we combine training and development dataset
        val dfT = if (config.testMode) dfT0.union(dfV) else dfT0
        val preprocessor = preprocess(dfT, s"${config.vocabPath}/${config.language}.vocab.txt")
        val (train, valid) = (preprocessor.transform(dfT), preprocessor.transform(dfV))
        train.show()
        println("#(train) = " + train.count())
        println("#(valid) = " + valid.count())
        val vocabulary = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
        println("LEFT 100 ==> ")
        println(vocabulary.take(100).mkString(", "))
        println("LAST 100 ==> ")
        println(vocabulary.takeRight(100).mkString(", "))
        println("#(vocab) = " + vocabulary.length)
        val labels = preprocessor.stages(0).asInstanceOf[StringIndexerModel].labelsArray(0)
        println(labels.mkString(":"))
        println("#(labels) = " + labels.length)
        // create a 1-based token map: token -> id
        val tokenMap = vocabulary.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
        val tokenIndex = tokenMap.map(p => (p._2, p._1))

        // multi-class evaluator
        val evaluator = new MulticlassClassificationEvaluator() // f1 is the default metric
        val tokenEmbeddingSize = if (config.pretrained) 300 else config.tokenEmbeddingSize
        config.mode match {
          case "predict" =>
            predict(config, tokenMap, labels, spark, preprocessor, None)
          case "train" =>
            config.modelType match {
              case "mlr" =>
                // create and train a classifier
                val classifier = new LogisticRegression().setFeaturesCol("features").setLabelCol("label").setElasticNetParam(1E-4)
                val model = classifier.fit(train)
                model.write.overwrite().save(s"${config.modelPath}/${config.modelType}/")
                // evaluate the model on the train and dev split
                val (uf, vf) = (model.transform(train).select("prediction", "label"), model.transform(valid).select("prediction", "label"))
                val f1U = evaluator.evaluate(uf)
                val f1V = evaluator.evaluate(vf)
                println(s"f1U = $f1U, f1V = $f1V")
                // write out scores
                val result = f"${config.language}%s;${config.modelType}%s;$f1U%4.3f;$f1V%4.3f\n"
                Files.write(Paths.get(config.scorePath), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
              case "rnn" =>
                // create a sequential model with random token embeddings
                val bigdl = Sequential()
                if (!config.pretrained) {
                  bigdl.add(Embedding(tokenMap.size + 1, config.tokenEmbeddingSize, inputLength = config.maxSeqLen))
                } else {
                  val embeddingFile = s"dat/emb/numberbatch-${config.language}-19.08.vocab.txt"
                  val vocab = Source.fromFile(s"dat/${config.language}.vocab.txt").getLines().toList
                  val wordIndex = vocab.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
                  bigdl.add(WordEmbeddingP(embeddingFile, wordIndex, inputLength = config.maxSeqLen))
                }
                // add multiple LSTM layers if config.layers > 1
                for (_ <- 0 until config.layers) {
                  bigdl.add(Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true)))
                }
                bigdl.add(Select(1, -1))
                bigdl.add(Dense(config.denseHiddenSize, activation = "relu"))
                bigdl.add(Dense(labels.length, activation = "softmax"))
                val (featureSize, labelSize) = (Array(Array(config.maxSeqLen)), Array(1))
                bigdl.summary()

                val (uf, vf) = bigdlPreprocess(config, tokenMap, train, valid)
                // create an estimator
                val estimator = if (config.weightedLoss) {
                  import spark.implicits._
                  val yf = uf.select("y").map(row => row.getDouble(0)).groupBy("value").count()
                  val labelFreq = yf.select("value", "count").collect().map(row => (row.getDouble(0), row.getLong(1)))
                  val total = labelFreq.map(_._2).sum.toFloat
                  val ws = labelFreq.map(p => (p._1, p._2 / total)).toMap
                  val tensor = Tensor(ws.size)
                  for (j <- ws.keySet)
                    tensor(j.toInt) = 1f/ws(j) // give higher weights to minority labels
                  NNEstimator(bigdl, ClassNLLCriterion(weights = tensor, sizeAverage = false, logProbAsInput = false, paddingValue = -1), featureSize, labelSize)
                } else NNEstimator(bigdl, ClassNLLCriterion(sizeAverage = false, logProbAsInput = false, paddingValue = -1), featureSize, labelSize)
                var logDir = s"sum/${config.language}"
                if (config.pretrained) logDir += "-x"
                val trainingSummary = TrainSummary(appName = config.modelType, logDir = logDir)
                val validationSummary = ValidationSummary(appName = config.modelType, logDir = logDir)
                estimator.setLabelCol("y").setFeaturesCol("x")
                  .setBatchSize(config.batchSize)
                  .setOptimMethod(new Adam(5E-5))
                  .setMaxEpoch(config.epochs)
                  .setTrainSummary(trainingSummary)
                  .setValidationSummary(validationSummary)
                  .setValidation(Trigger.everyEpoch, vf, Array(new Loss(), new Top1Accuracy()), config.batchSize)
                // train
                estimator.fit(uf)
                // save the model
                var modelPath = s"${config.modelPath}/rnn/${config.language}-${config.modelType}"
                if (config.pretrained) modelPath += "-x"
                bigdl.saveModel(modelPath, overWrite = true)
                val (ufZ, vfZ) = eval(config, uf, vf, Some(bigdl))
                // evaluation
                val af = ufZ.withColumn("prediction", argmax(col("z")))
                val bf = vfZ.withColumn("prediction", argmax(col("z")))
                af.show(false)
                val f1U = evaluator.evaluate(af.select("prediction", "label"))
                val f1V = evaluator.evaluate(bf.select("prediction", "label"))
                println(s"f1U = $f1U, f1V = $f1V")
                // write out scores
                val result = f"${config.language}%s;${config.modelType}%s;${config.layers}%d;$tokenEmbeddingSize%d;${config.tokenHiddenSize}%d;${config.denseHiddenSize}%d;$f1U%4.3f;$f1V%4.3f\n"
                Files.write(Paths.get(config.scorePath), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
                println("0 predictions = " + bf.filter(col("prediction") === 0).count())
                // run on the test data
                predict(config, tokenMap, labels, spark, preprocessor, Some(bigdl))
              case "jsl" =>
                // train a pipeline on dfT
                val pipelineModel = jslTrain(config, dfT)
                // predict
                jslTest(spark, config, Some(pipelineModel))
              case _ => println("Unsupported model type.")
            }
          case "eval" =>
            val (uf, vf) = bigdlPreprocess(config, tokenMap, train, valid)
            val (ufZ, vfZ) = eval(config, uf, vf, None)
            vfZ.show(false)
            val af = ufZ.withColumn("prediction", argmax(col("z")))
            val bf = vfZ.withColumn("prediction", argmax(col("z")))
            af.show(false)
            val f1U = evaluator.evaluate(af.select("prediction", "label"))
            val f1V = evaluator.evaluate(bf.select("prediction", "label"))
            println(s"f1U = $f1U, f1V = $f1V")
            // write out scores
            val result = f"${config.language}%s;${config.modelType}%s;${config.layers}%d;$tokenEmbeddingSize%d;${config.tokenHiddenSize}%d;${config.denseHiddenSize}%d;$f1U%4.3f;$f1V%4.3f\n"
            Files.write(Paths.get(config.scorePath), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
            println("0 predictions = " + bf.filter(col("prediction") === 0).count())
          case "evalJSL" =>
            jslTest(spark, config, None)
          case "experiment" =>
            // post shared task experiments for manuscript
            val (featureSize, labelSize) = (Array(Array(config.maxSeqLen)), Array(1))
            val (uf, vf) = bigdlPreprocess(config, tokenMap, train, valid)
            val ws = Array(64, 100, 200, 300)
            val hs = Array(100, 200, 300)
            for (w <- ws) {
              for (h <- hs) {
                for (_ <- 0 until 3) { // run each configuration three times
                  val bigdl = Sequential()
                  bigdl.add(Embedding(tokenMap.size + 1, w, inputLength = config.maxSeqLen))
                  // add multiple LSTM layers if config.layers > 1
                  for (_ <- 0 until config.layers) {
                    bigdl.add(Bidirectional(LSTM(outputDim = h, returnSequences = true)))
                  }
                  bigdl.add(Select(1, -1))
                  bigdl.add(Dense(config.denseHiddenSize, activation = "relu"))
                  bigdl.add(Dense(labels.length, activation = "softmax"))

                  // create an estimator
                  val estimator = NNEstimator(bigdl, ClassNLLCriterion(sizeAverage = false, logProbAsInput = false, paddingValue = -1), featureSize, labelSize)
                  var logDir = s"sum/${config.language}"
                  if (config.pretrained) logDir += "-x"
                  val trainingSummary = TrainSummary(appName = config.modelType, logDir = logDir)
                  val validationSummary = ValidationSummary(appName = config.modelType, logDir = logDir)
                  estimator.setLabelCol("y").setFeaturesCol("x")
                    .setBatchSize(config.batchSize)
                    .setOptimMethod(new Adam(5E-5))
                    .setMaxEpoch(config.epochs)
                    .setTrainSummary(trainingSummary)
                    .setValidationSummary(validationSummary)
                    .setValidation(Trigger.everyEpoch, vf, Array(new Loss(), new Top1Accuracy()), config.batchSize)
                  // train
                  estimator.fit(uf)
                  val (ufZ, vfZ) = eval(config, uf, vf, Some(bigdl))
                  // evaluate on the training set and development set
                  val af = ufZ.withColumn("prediction", argmax(col("z")))
                  val bf = vfZ.withColumn("prediction", argmax(col("z")))
                  val f1U = evaluator.evaluate(af.select("prediction", "label"))
                  val f1V = evaluator.evaluate(bf.select("prediction", "label"))
                  // write out scores
                  val result = f"${config.language}%s;${config.modelType}%s;${config.layers}%d;$w%d;$h%d;${config.denseHiddenSize}%d;$f1U%4.3f;$f1V%4.3f\n"
                  Files.write(Paths.get(config.scorePath), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
                }
              }
            }
          case "experimentConceptNet" =>
            // post shared task experiments for manuscript
            val (featureSize, labelSize) = (Array(Array(config.maxSeqLen)), Array(1))
            val (uf, vf) = bigdlPreprocess(config, tokenMap, train, valid)
            val hs = Array(100, 200, 256, 300)
            val embeddingFile = s"dat/emb/numberbatch-${config.language}-19.08.vocab.txt"
            val vocab = Source.fromFile(s"dat/${config.language}.vocab.txt").getLines().toList
            val wordIndex = vocab.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
            val w = 300
            for (h <- hs) {
              for (_ <- 0 until 3) { // run each configuration three times
                val bigdl = Sequential()
                bigdl.add(WordEmbeddingP(embeddingFile, wordIndex, inputLength = config.maxSeqLen))
                // add multiple LSTM layers if config.layers > 1
                for (_ <- 0 until config.layers) {
                  bigdl.add(Bidirectional(LSTM(outputDim = h, returnSequences = true)))
                }
                bigdl.add(Select(1, -1))
                bigdl.add(Dense(config.denseHiddenSize, activation = "relu"))
                bigdl.add(Dense(labels.length, activation = "softmax"))

                // create an estimator
                val estimator = NNEstimator(bigdl, ClassNLLCriterion(sizeAverage = false, logProbAsInput = false, paddingValue = -1), featureSize, labelSize)
                var logDir = s"sum/${config.language}"
                if (config.pretrained) logDir += "-x"
                val trainingSummary = TrainSummary(appName = config.modelType, logDir = logDir)
                val validationSummary = ValidationSummary(appName = config.modelType, logDir = logDir)
                estimator.setLabelCol("y").setFeaturesCol("x")
                  .setBatchSize(config.batchSize)
                  .setOptimMethod(new Adam(5E-5))
                  .setMaxEpoch(config.epochs)
                  .setTrainSummary(trainingSummary)
                  .setValidationSummary(validationSummary)
                  .setValidation(Trigger.everyEpoch, vf, Array(new Loss(), new Top1Accuracy()), config.batchSize)
                // train
                estimator.fit(uf)
                val (ufZ, vfZ) = eval(config, uf, vf, Some(bigdl))
                // evaluate on the training set and development set
                val af = ufZ.withColumn("prediction", argmax(col("z")))
                val bf = vfZ.withColumn("prediction", argmax(col("z")))
                val f1U = evaluator.evaluate(af.select("prediction", "label"))
                val f1V = evaluator.evaluate(bf.select("prediction", "label"))
                // write out scores
                val result = f"${config.language}%s;${config.modelType}%s-N;${config.layers}%d;$w%d;$h%d;${config.denseHiddenSize}%d;$f1U%4.3f;$f1V%4.3f\n"
                Files.write(Paths.get(config.scorePath), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
              }
            }
        }
        spark.stop()
      case None => println("Invalid config!")
    }
  }
}
