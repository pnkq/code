package vlp.woz.nlu

import com.intel.analytics.bigdl.dllib.keras.{Model, Sequential}
import com.intel.analytics.bigdl.dllib.keras.layers.{BERT, Bidirectional, Dense, Embedding, Input, InputLayer, KerasLayerWrapper, LSTM, Merge, RepeatVector, Reshape, Select, SelectTable, SplitTensor, Squeeze, ZeroPadding1D}
import com.intel.analytics.bigdl.dllib.keras.models.{KerasNet, Models}
import com.intel.analytics.bigdl.dllib.nn.{ClassNLLCriterion, TimeDistributedCriterion, Transpose}
import com.intel.analytics.bigdl.dllib.nnframes.{NNEstimator, NNModel}
import com.intel.analytics.bigdl.dllib.optim.{Adam, Trigger}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.{Engine, Shape}
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.linalg.DenseVector
import scopt.OptionParser
import vlp.woz.Sequencer4BERT
import vlp.woz.act.Act

import java.nio.file.{Files, Paths, StandardOpenOption}

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
    * <li>utterance: String, is an original text</li>
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

  private def createEncoderLSTM(numTokens: Int, numEntities: Int, config: ConfigNLU): Sequential[Float] = {
    val sequential = Sequential[Float]()
    sequential.add(InputLayer[Float](inputShape = Shape(config.maxSeqLen)))
    sequential.add(Embedding[Float](inputDim = numTokens, outputDim = config.embeddingSize))
    for (j <- 0 until config.numLayers)
      sequential.add(Bidirectional[Float](LSTM[Float](outputDim = config.recurrentSize, returnSequences = true)))
    sequential.add(Dense[Float](numEntities, activation = "softmax"))
    sequential
  }

  private def createEncoderBERT(numTokens: Int, numEntities: Int, config: ConfigNLU): KerasNet[Float] = {
    val input = Input[Float](inputShape = Shape(4*config.maxSeqLen), name = "input")
    val reshape = Reshape[Float](targetShape = Array(4, config.maxSeqLen)).inputs(input)
    val split = SplitTensor[Float](1, 4).inputs(reshape)
    val selectIds = SelectTable[Float](0).setName("inputId").inputs(split)
    val inputIds = Squeeze[Float](1).inputs(selectIds)
    val selectSegments = SelectTable[Float](1).setName("segmentId").inputs(split)
    val segmentIds = Squeeze[Float](1).inputs(selectSegments)
    val selectPositions = SelectTable[Float](2).setName("positionId").inputs(split)
    val positionIds = Squeeze[Float](1).inputs(selectPositions)
    val selectMasks = SelectTable[Float](3).setName("masks").inputs(split)
    val masksReshaped = Reshape[Float](targetShape = Array(1, 1, config.maxSeqLen)).setName("maskReshape").inputs(selectMasks)

    // use a BERT layer, not output all blocks (there will be 2 outputs)
    val bert = BERT[Float](numTokens, config.embeddingSize, config.numLayers, config.numHeads, config.maxSeqLen, config.hiddenSize, outputAllBlock = false).setName("BERT")
    val bertNode = bert.inputs(Array(inputIds, segmentIds, positionIds, masksReshaped))
    // get the pooled output which processes the hidden state of the last layer with regard to the first
    //  token of the sequence. This would be useful for classification tasks.
//    val lastState = SelectTable[Float](1).setName("firstBlock").inputs(bertNode)
    // get all BERT states
    val bertOutput = SelectTable[Float](0).setName("bertOutput").inputs(bertNode)
    val output = Dense[Float](numEntities, activation = "softmax").setName("output").inputs(bertOutput)
    Model[Float](input, output)
  }

  private def createJointEncoderLSTM(numTokens: Int, numEntities: Int, numActs: Int, config: ConfigNLU): KerasNet[Float] = {
    // (1) first part: 2 layers BiLSTM
    val input = Input[Float](inputShape = Shape(config.maxSeqLen))
    val embeddings = Embedding[Float](inputDim = numTokens, outputDim = config.embeddingSize).inputs(input)
    val rnn1 = Bidirectional[Float](LSTM[Float](outputDim = config.recurrentSize, returnSequences = true)).inputs(embeddings)
    val rnn2 = Bidirectional[Float](LSTM[Float](outputDim = config.recurrentSize, returnSequences = true)).inputs(rnn1)
    val entityOutput = Dense[Float](numEntities, activation = "softmax").inputs(rnn2)
    // (2) second part: sigmoid activation for each act prediction
    // select the last hidden state of rnn2 as the representation for act prediction
    val actDense = Select[Float](1, -1).inputs(rnn2)
    val actOutput = Dense[Float](numActs, activation = "sigmoid").inputs(actDense)
    // duplicate the actOutput [a] => [a, a]
    val duplicate = RepeatVector[Float](2).inputs(actOutput)
    // right pad the entity output
    val transposeEntity = new KerasLayerWrapper(new Transpose[Float](permutations = Array((2, 3)))).inputs(entityOutput)
    val zeroPaddingEntity = new ZeroPadding1D[Float](padding = Array(0, numActs)).inputs(transposeEntity)
    val transposeBackEntity = new KerasLayerWrapper(new Transpose[Float](permutations = Array((2, 3)))).inputs(zeroPaddingEntity)
    // left pad the act output
    val reshapeActOutput = new Reshape[Float](targetShape = Array(2, numActs)).inputs(duplicate)
    val transposeAct = new KerasLayerWrapper(new Transpose[Float](permutations = Array((2, 3)))).inputs(reshapeActOutput)
    val zeroPaddingAct = new ZeroPadding1D[Float](padding = Array(numEntities, 0)).inputs(transposeAct)
    val transposeBackAct = new KerasLayerWrapper(new Transpose[Float](permutations = Array((2, 3)))).inputs(zeroPaddingAct)
    // concat the actOutput with the entityOutput. The output shape should be (maxSeqLen, numEntities + numActs)
    val merge = Merge.merge(inputs = List(transposeBackEntity, transposeBackAct), mode = "concat", concatAxis = 1) // concat along the temporal dimension
    Model(input, merge)
  }

  private def predict(encoder: KerasNet[Float], vf: DataFrame, featuresCol: String = "features", argmax: Boolean=true): DataFrame = {
    val sequential = Sequential[Float]()
    sequential.add(encoder)
    // bigdl produces 3-d output results (including batch dimension), we need to convert it to 2-d results.
    if (argmax)
      sequential.add(ArgMaxLayer[Float]())
    sequential.summary()
    // pass to a Spark model and run prediction
    val model = NNModel[Float](sequential).setFeaturesCol(featuresCol)
    model.transform(vf)
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
   * Exports result data frame (2-col format) into a text file of CoNLL-2003 format for
   * evaluation with CoNLL evaluation script ("target <space> prediction").
   * @param result a data frame of two columns "prediction, target"
   * @param config configuration
   * @param split a split name
   */
  private def export(result: DataFrame, config: ConfigNLU, split: String) = {
    val spark = SparkSession.getActiveSession.get
    import spark.implicits._
    val ss = result.map { row =>
      val prediction = row.getSeq[String](0)
      val target = row.getSeq[String](1)
      val lines = target.zip(prediction).map(p => p._1 + " " + p._2)
      lines.mkString("\n") + "\n"
    }.collect()
    val s = ss.mkString("\n")
    Files.write(Paths.get(s"dat/woz/nlu/${config.modelType}-${config.numLayers}-$split.txt"), s.getBytes, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }

  private def evaluateAct(spark: SparkSession, result: DataFrame): Double = {
    import spark.implicits._
    val df = result.map { row =>
      val ys = row.getAs[DenseVector](0).toArray.takeRight(2).filter(_ >= 0)
      val zs = row.getAs[Seq[Float]](1).toArray.takeRight(ys.length).map(_.toDouble)
      (ys, zs)
    }.toDF("label", "prediction")
    df.show(false)
    val evaluator = new MultilabelClassificationEvaluator().setMetricName("accuracy")
    evaluator.evaluate(df)
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
      opt[Double]('a', "learningRate").action((x, conf) => conf.copy(learningRate = x)).text("learning rate")
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
        val modelPath = s"bin/woz-${config.modelType}-${config.numLayers}.bigdl"
        val (featuresCol, featureSize) = config.modelType match {
          case "lstm" => ("features", Array(config.maxSeqLen))
          case "bert" => ("featuresBERT", Array(4*config.maxSeqLen))
          case "join" => ("features", Array(config.maxSeqLen))
        }

        config.mode match {
          case "init" =>
            saveDatasets(spark)
            val df = spark.read.json("dat/woz/nlu/train")
            preprocess(df, s"$basePath/pre")
          case "train" =>
            val preprocessor = PipelineModel.load(s"$basePath/pre")
            val vocab = preprocessor.stages(0).asInstanceOf[CountVectorizerModel].vocabulary
            val entities = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
            val acts = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
            // BigDL uses 1-based index for targets (entity, act). For token embedding, we can use 0-based index.
            val vocabDict = vocab.zipWithIndex.map(p => (p._1, p._2)).toMap
            val entityDict = entities.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
            // act indices are shifted by length(entities)
            val actDict = acts.zipWithIndex.map(p => (p._1, p._2 + 1 + entities.length)).toMap

            val sequencerTokens = new Sequencer(vocabDict, config.maxSeqLen, 0f).setInputCol("tokens").setOutputCol("features")
            val sequencerEntities = new Sequencer(entityDict, config.maxSeqLen, -1f).setInputCol("slots").setOutputCol("slotIdx")
            // most utterances have at most 2 acts; so a 2-dimension vector for act encoding is sufficient.
            val sequencerActs = new Sequencer(actDict, 2, -1f).setInputCol("actNames").setOutputCol("actIdx")

            val (train, dev) = (spark.read.json("dat/woz/nlu/train"), spark.read.json("dat/woz/nlu/dev"))
            // remove samples which are longer than maxSeqLen
            val (trainDF, devDF) = (
              train.withColumn("n", size(col("tokens"))).filter(col("n") <= config.maxSeqLen),
              dev.withColumn("n", size(col("tokens"))).filter(col("n") <= config.maxSeqLen)
            )
            val (uf, vf) = config.modelType match {
              case "lstm" =>
                (
                  sequencerTokens.transform(sequencerEntities.transform(trainDF)),
                  sequencerTokens.transform(sequencerEntities.transform(devDF))
                )
              case "bert" =>
                val sequencerBERT = new Sequencer4BERT(vocabDict, config.maxSeqLen, 0f).setInputCol("tokens").setOutputCol("featuresBERT")
                (
                  sequencerBERT.transform(sequencerTokens.transform(sequencerEntities.transform(trainDF))),
                  sequencerBERT.transform(sequencerTokens.transform(sequencerEntities.transform(devDF)))
                )
              case "join" =>
                // shift the act indices by numEntities
                val shift = udf((xs: Seq[Int]) => xs.map(_ + entities.length))
                val (pf, qf) = (
                  sequencerActs.transform(trainDF).withColumn("actIdxShifted", col("actIdx")),
                  sequencerActs.transform(devDF).withColumn("actIdxShifted", col("actIdx"))
                )
                val assembler = new VectorAssembler().setInputCols(Array("slotIdx", "actIdxShifted")).setOutputCol("label")
                (
                  assembler.transform(sequencerTokens.transform(sequencerEntities.transform(pf))),
                  assembler.transform(sequencerTokens.transform(sequencerEntities.transform(qf)))
                )
            }
            // save uf and vf for other modes (eval/predict)
            uf.repartition(2).write.mode("overwrite").parquet("dat/woz/nlu/uf")
            vf.repartition(1).write.mode("overwrite").parquet("dat/woz/nlu/vf")

            val encoder = config.modelType match {
              case "lstm" => createEncoderLSTM(vocab.length, entities.length, config)
              case "bert" => createEncoderBERT(vocab.length, entities.length, config)
              case "join" => createJointEncoderLSTM(vocab.length, entities.length, acts.length, config)
            }
            encoder.summary()

            val (labelCol, labelSize) = if (config.modelType == "join")
              ("label", Array(config.maxSeqLen + 2))
            else ("slotIdx", Array(config.maxSeqLen))

            val w = if (config.modelType == "join") {
              val w1 = labelWeights(spark, uf, "slotIdx").toArray().map(_ * config.lambdaSlot)
              val w2 = labelWeights(spark, uf, "actIdx").toArray().map(_ * config.lambdaAct)
              Tensor[Float](w1 ++ w2, Array(w1.length + w2.length))
            } else labelWeights(spark, uf, "slotIdx")
            println(w)

            val criterion = ClassNLLCriterion[Float](weights = w, sizeAverage = false, paddingValue = -1)
            val estimator = NNEstimator(encoder, TimeDistributedCriterion(criterion, sizeAverage = true), featureSize, labelSize)
            val trainingSummary = TrainSummary(appName = config.modelType, logDir = "sum/woz/")
            val validationSummary = ValidationSummary(appName = config.modelType, logDir = "sum/woz/")
            val batchSize = if (config.batchSize % numCores != 0) numCores * 4; else config.batchSize
            uf.select(featuresCol, labelCol).show(false)
            estimator.setLabelCol(labelCol).setFeaturesCol(featuresCol)
              .setBatchSize(batchSize)
              .setOptimMethod(new Adam[Float](config.learningRate))
              .setMaxEpoch(config.epochs)
              .setTrainSummary(trainingSummary)
              .setValidationSummary(validationSummary)
              .setValidation(Trigger.everyEpoch, vf, Array(new TimeDistributedTop1Accuracy(paddingValue = -1)), batchSize)
            estimator.fit(uf)

            encoder.saveModel(modelPath, overWrite = true)

            // predict and export results
            val pf = predict(encoder, uf, featuresCol)
            val qf = predict(encoder, vf, featuresCol)
            // extract act prediction for the joint model:
            if (config.modelType == "join") {
              println("Train multi-label performance (f1Measure): ", evaluateAct(spark, pf.select("label", "prediction")))
              println("Valid multi-label performance (f1Measure): ", evaluateAct(spark, qf.select("label", "prediction")))
            }
            // convert "prediction" column to human-readable label column "zs"
            val labelDict = entityDict.map { case (k, v) => (v.toDouble, k) }
            val sequencer = new SequencerDouble(labelDict).setInputCol("prediction").setOutputCol("zs")
            val af = sequencer.transform(pf)
            val bf = sequencer.transform(qf)
            // slot prediction as CoNLL format files
            export(af.select("zs", "slots"), config, "train")
            export(bf.select("zs", "slots"), config, "valid")

          case "eval" =>
            val encoder =  Models.loadModel[Float](modelPath)
            encoder.summary()
            val uf = spark.read.parquet("dat/woz/nlu/uf")
            val vf = spark.read.parquet("dat/woz/nlu/vf")
            // predict and export results
            val preprocessor = PipelineModel.load(s"$basePath/pre")
            val pf = predict(encoder, uf, featuresCol)
            val qf = predict(encoder, vf, featuresCol)
            qf.select("label", "prediction").show(false)
            if (config.modelType == "join") {
              println("Train multi-label performance (f1Measure): ", evaluateAct(spark, pf.select("label", "prediction")))
              println("Valid multi-label performance (f1Measure): ", evaluateAct(spark, qf.select("label", "prediction")))
            }
            // convert "prediction" column to human-readable label column "zs"
            val entities = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
            val entityDict = entities.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
            val labelDict = entityDict.map { case (k, v) => (v.toDouble, k) }
            val sequencer = new SequencerDouble(labelDict).setInputCol("prediction").setOutputCol("zs")
            val af = sequencer.transform(pf)
            val bf = sequencer.transform(qf)
            export(af.select("zs", "slots"), config, "train")
            export(bf.select("zs", "slots"), config, "valid")
          case "join" =>
            val encoder = createJointEncoderLSTM(100, 50, 30, config)
            encoder.summary()
        }
        spark.stop()
      case None =>
    }
  }
}
