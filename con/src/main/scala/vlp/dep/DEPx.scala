package vlp.dep

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.keras.models.{Model, Models, KerasNet}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.{ClassNLLCriterion, TimeDistributedMaskCriterion}
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.{Trigger, Loss}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import scopt.OptionParser
import vlp.ner.{ArgMaxLayer, Sequencer, TimeDistributedTop1Accuracy}

import java.nio.file.{Files, Paths, StandardOpenOption}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window


/**
* phuonglh, April 2024.
*/

object DEPx {

  /**
   * Linearize a graph into multiple sequences: Seq[word], Seq[PoS], Seq[uPos], Seq[fs], Seq[labels], and Seq[offsets].
   * @param graph a dependency graph
   * @param las labeled attachment score
   * @return a sequence of sequences.
   */
  private def linearize(graph: Graph, las: Boolean = false): Seq[Seq[String]] = {
    val tokens = graph.sentence.tokens.tail // remove the ROOT token at the beginning
    val words = tokens.map(_.word.toLowerCase()) // make all token lowercase
    val partsOfSpeech = tokens.map(_.partOfSpeech)
    val featureStructure = tokens.map(_.featureStructure)
    val uPoS = tokens.map(_.universalPartOfSpeech)
    val labels = tokens.map(_.dependencyLabel)
    // compute the offset value to the head for each token
    val n = graph.sentence.tokens.size
    val offsets = try {
      tokens.map { token =>
        var o = if (token.head.toInt == 0) 0 else token.head.toInt - token.id.toInt
        if (Math.abs(o) > n) o = 1
        o.toString
      }
    } catch {
      case _: NumberFormatException =>
        print(graph)
        Seq.empty[String]
    } finally {
      Seq.empty[String]
    }
    val offsetLabels = if (las) {
      offsets.zip(labels).map(pair => pair._1 + ":" + pair._2)
    } else offsets
    Seq(words, partsOfSpeech, uPoS, featureStructure, labels, offsetLabels)
  }

  /**
   * Read graphs from a corpus, filter too long or too short graphs and convert them to a df.
   * @param spark Spark session
   * @param path path to a UD treebank
   * @param maxSeqLen maximum sequence length
   * @param las LAS or UAS
   * @return a data frame
   */
  def readGraphs(spark: SparkSession, path: String, maxSeqLen: Int, las: Boolean = false): DataFrame = {
    // read graphs and remove too-long or two-short sentences
    // val graphs = GraphReader.read(path).filter(_.sentence.tokens.size <= maxSeqLen).filter(_.sentence.tokens.size >= 5)
    // read graphs and split long graphs if necessary
    val graphs = GraphReader.read(path).filter(_.sentence.tokens.size >= 5).flatMap { graph => 
      if (graph.sentence.tokens.size <= maxSeqLen) Array(graph) else {
        val (left, right) = graph.splitGraph(graph)
        val (m, n) = (left.sentence.tokens.size, right.sentence.tokens.size)
        if (m <= maxSeqLen && n <= maxSeqLen) {
          Array(left, right) 
        } else if (m > maxSeqLen && n <= maxSeqLen) {
          Array(right)
        } else if (m <= maxSeqLen && n > maxSeqLen) {
          Array(left)
        } else Array.empty[Graph]
      }
    }
    // linearize the graph
    val xs = graphs.map { graph => Row(linearize(graph, las): _*) } // need to scroll out the parts with :_*
    val schema = StructType(Array(
      StructField("tokens", ArrayType(StringType, containsNull = true)),
      StructField("partsOfSpeech", ArrayType(StringType, containsNull = true)),
      StructField("uPoS", ArrayType(StringType, containsNull = true)),
      StructField("featureStructure", ArrayType(StringType, containsNull = true)),
      StructField("labels", ArrayType(StringType, containsNull = true)),
      StructField("offsets", ArrayType(StringType, containsNull = true))
    ))
    spark.createDataFrame(spark.sparkContext.parallelize(xs), schema)
  }

  /**
   * Create a preprocessing pipeline.
   *
   * @param df     a data frame
   * @param config configuration
   * @return a pipeline model
   */
  private def createPipeline(df: DataFrame, config: ConfigDEP): PipelineModel = {
    val vectorizerOffsets = new CountVectorizer().setInputCol("offsets").setOutputCol("off")
    val vectorizerTokens = new CountVectorizer().setInputCol("tokens").setOutputCol("tok").setVocabSize(config.maxVocabSize)
    val vectorizerPartsOfSpeech = new CountVectorizer().setInputCol("uPoS").setOutputCol("pos") 
    val vectorizerFeatureStructure = new CountVectorizer().setInputCol("featureStructure").setOutputCol("fst")
    val stages = Array(vectorizerOffsets, vectorizerTokens, vectorizerPartsOfSpeech, vectorizerFeatureStructure)
    val pipeline = new Pipeline().setStages(stages)
    val model = pipeline.fit(df)
    model.write.overwrite().save(s"${config.modelPath}/${config.language}-pre")
    val ef = model.transform(df)
    ef.repartition(1).write.mode("overwrite").parquet(s"${config.modelPath}/${config.language}-dfs")
    model
  }

  /**
   * Evaluate a model on training data frame (uf) and validation data frame (vf) which have been preprocessed
   * by the training pipeline.
   * @param bigdl the model
   * @param config config
   * @param uf training df
   * @param vf validation df
   * @param featureColNames feature column names
   */
  def eval(bigdl: KerasNet[Float], config: ConfigDEP, uf: DataFrame, vf: DataFrame, featureColNames: Array[String]): Seq[Double] = {
    // create a sequential model and add a custom ArgMax layer at the end of the model
    val sequential = Sequential()
    sequential.add(bigdl)
    // bigdl produces 3-d output results (including batch dimension), we need to convert it to 2-d results.
    sequential.add(ArgMaxLayer())
    // run prediction on the training set and validation set
    val predictions = Array(
      sequential.predict(uf, featureCols = featureColNames, predictionCol = "z"),
      sequential.predict(vf, featureCols = featureColNames, predictionCol = "z")
    )
    sequential.summary()
    val spark = SparkSession.getActiveSession.get
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("y").setPredictionCol("z").setMetricName("accuracy")
    import spark.implicits._
    val scores = for (prediction <- predictions) yield {
      val zf = prediction.select("o", "z").map { row =>
        val o = row.getAs[Vector](0).toArray.filter(_ >= 0)
        val p = row.getSeq[Float](1).take(o.size)
        (o, p)
      }
      // show the prediction, each row is a graph
      zf.toDF("offsets", "prediction").show(5, false)
      // flatten the prediction, convert to double for evaluation using Spark lib
      val yz = zf.flatMap(p => p._1.zip(p._2.map(_.toDouble))).toDF("y", "z")
      yz.show(15)
      evaluator.evaluate(yz)
    }
    scores
  }

  def parseOptions: OptionParser[ConfigDEP] = {
    new OptionParser[ConfigDEP](getClass.getName) {
      head(getClass.getName, "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 16g")
      opt[String]('E', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 16g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/predict")
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language (eng/ind/vie)")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('n', "maxSeqLength").action((x, conf) => conf.copy(maxSeqLen = x)).text("max sequence length")
      opt[Int]('j', "layers").action((x, conf) => conf.copy(layers = x)).text("number of RNN layers or Transformer blocks")
      opt[Int]('u', "heads").action((x, conf) => conf.copy(heads = x)).text("number of Transformer heads")
      opt[Int]('w', "tokenEmbeddingSize").action((x, conf) => conf.copy(tokenEmbeddingSize = x)).text("token embedding size")
      opt[Int]('h', "tokenHiddenSize").action((x, conf) => conf.copy(tokenHiddenSize = x)).text("encoder hidden size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x)).text("learning rate, default value is 5E-3")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model folder, default is 'bin/'")
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path")
      opt[String]('s', "scorePath").action((x, conf) => conf.copy(scorePath = x)).text("score path")
      opt[String]('L', "las").action((_, conf) => conf.copy(las = true)).text("labeled parsing")
    }
  }

  def createSequencerX(spark: SparkSession, config: ConfigDEP) = {
    val graphU = spark.read.parquet(s"dat/dep/${config.language}-graphx-train")
    val graphV = spark.read.parquet(s"dat/dep/${config.language}-graphx-dev")
    import spark.implicits._
    val graphUMap = graphU.map { row => (row.getString(0), row.getAs[Seq[Double]](1)) }.collect().toMap
    val graphVMap = graphV.map { row => (row.getString(0), row.getAs[Seq[Double]](1)) }.collect().toMap
    // join two maps
    val graphMap = graphUMap ++ graphVMap
    val zipFunc = udf((a: Seq[String], b: Seq[String]) => {a.zip(b).map(p => p._1 + ":" + p._2) }) // t:p = tokens:uPoS
    // create a sequencer for graphX features
    val sequencerX1 = new SequencerX(graphMap, config.maxSeqLen, 3).setInputCol("t:p").setOutputCol("xs1")
    // flatten the "xs1" column to get a vector x1 (of numberOfNetworkXFeatures * maxSeqLen elements)
    val flattenFunc = udf((xs: Seq[Vector]) => { Vectors.dense(xs.flatMap(v => v.toArray).toArray) })

    // extended features (32-dim embedding of Node2Vec)
    // read Node2Vec features from the training split (-nodeId.txt and -nodeVec.txt)
    val graphNodeId = spark.read.csv(s"dat/dep/${config.language}-nodeId.txt").toDF("t:p")
    val graphNodeVec = spark.read.csv(s"dat/dep/${config.language}-nodeVec.txt").toDF("s")
    // convert "s" string column to a Seq[Double] column
    val splitFunc = udf((s: String) => s.split(" ").map(_.toDouble))
    val graphNodeVecSeq = graphNodeVec.withColumn("z", splitFunc(col("s")))
    // add an index column to two dfs 
    val windowSpec = Window.orderBy(monotonically_increasing_id())
    val graphNodeIdWithIndex = graphNodeId.withColumn("id", row_number().over(windowSpec))
    val graphNodeVecSeqWithIndex = graphNodeVecSeq.withColumn("id", row_number().over(windowSpec))
    // now join two dfs by the index column
    val graphNode = graphNodeIdWithIndex.join(graphNodeVecSeqWithIndex, Seq("id")).select("t:p", "z")
    // convert to a map
    val graphNodeMap = graphNode.map { row => (row.getString(0), row.getAs[Seq[Double]](1)) }.collect().toMap
    // create a sequencer for Node2Vec features
    val sequencerX2 = new SequencerX(graphNodeMap, config.maxSeqLen, 32).setInputCol("t:p").setOutputCol("xs2")

    (zipFunc, sequencerX1, flattenFunc, sequencerX2)
  }

  def addNodeEmbeddingFeatures(spark: SparkSession, config: ConfigDEP, dfs: Array[DataFrame]): Array[DataFrame] = {
    val (zipFunc, sequencerX1, flattenFunc, sequencerX2) = createSequencerX(spark, config)
    dfs.map { df => 
      val gfx = df.withColumn("t:p", zipFunc(col("tokens"), col("uPoS")))
      val gfy = sequencerX1.transform(gfx)
      // flatten the "xs1" column to get a vector x1 (of numberOfNetworkXFeatures * maxSeqLen elements)
      val gfz = gfy.withColumn("x1", flattenFunc(col("xs1")))
      val gfy2 = sequencerX2.transform(gfz)
      // flatten the "xs2" column to get a vector x2 (of numberOfNode2VecFeatures * maxSeqLen elements)
      val gfz2 = gfy2.withColumn("x2", flattenFunc(col("xs2")))
      // assemble the input vectors into one 
      val assembler = config.modelType match {
        case "b" => new VectorAssembler().setInputCols(Array("b", "x1", "x2")).setOutputCol("b+x")
        case "bx" => new VectorAssembler().setInputCols(Array("b", "p", "f", "x1", "x2")).setOutputCol("b+p+f+x")
        case _ => new VectorAssembler().setInputCols(Array("t", "p", "f", "x1", "x2")).setOutputCol("t+p+f+x")
      }
      assembler.transform(gfz2)
    }
  }

  // user-defined function for creating BERT input from a token index vector
  val createInputBERT = udf((x: Vector) => {
    val v = x.toArray // x is a dense vector (produced by a Sequencer)
    // token type, all are 0 (0 for sentence A, 1 for sentence B -- here we have only one sentence)
    val types: Array[Double] = Array.fill[Double](v.length)(0)
    // positions, start from 0
    val positions = Array.fill[Double](v.length)(0)
    for (j <- v.indices)
      positions(j) = j
    // attention mask with indices in [0, 1]
    // It's a mask to be used if the input sequence length is smaller than maxSeqLen
    // find the last non-zero element index
    var n = v.length - 1
    while (n >= 0 && v(n) == 0) n = n - 1
    val masks = Array.fill[Double](v.length)(1)
    // padded positions have a mask value of 0 (which are not attended to)
    for (j <- n + 1 until v.length) {
      masks(j) = 0
    }
    Vectors.dense(v ++ types ++ positions ++ masks)
  })
  

  def main(args: Array[String]): Unit = {
    parseOptions.parse(args, ConfigDEP()) match {
      case Some(config) =>
        val conf = new SparkConf().setAppName(getClass.getName).setMaster(config.master)
          .set("spark.driver.memory", config.driverMemory)
          .set("spark.executor.memory", config.executorMemory)
        // Creates or gets SparkContext with optimized configuration for BigDL performance.
        // The method will also initialize the BigDL engine.
        val sc = NNContext.initNNContext(conf)
        sc.setLogLevel("ERROR")
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
        // determine the training and validation paths
        val (trainPath, validPath, testPath, gloveFile, numberbatchFile) = config.language match {
          case "eng" => ("dat/dep/UD_English-EWT/en_ewt-ud-train.conllu",
            "dat/dep/UD_English-EWT/en_ewt-ud-dev.conllu",
            "dat/dep/UD_English-EWT/en_ewt-ud-test.conllu",
            "dat/emb/glove.6B.100d.vocab.txt",
            "dat/emb/numberbatch-en-19.08.vocab.txt")
          case "fra" => ("dat/dep/UD_French-GSD/fr_gsd-ud-train.conllu",
            "dat/dep/UD_French-GSD/fr_gsd-ud-dev.conllu",
            "dat/dep/UD_French-GSD/fr_gsd-ud-test.conllu",
            "dat/emb/cc.fr.300.vocab.vec",
            "dat/emb/numberbatch-fr-19.08.vocab.txt")            
          case "ind" => ("dat/dep/UD_Indonesian-GSD/id_gsd-ud-train.conllu",
            "dat/dep/UD_Indonesian-GSD/id_gsd-ud-dev.conllu",
            "dat/dep/UD_Indonesian-GSD/id_gsd-ud-test.conllu",
            "dat/emb/cc.id.300.vocab.vec",
            "dat/emb/numberbatch-id-19.08.vocab.txt")
          case "vie" => ("dat/dep/UD_Vietnamese-VTB/vi_vtb-ud-train.conllu",
            "dat/dep/UD_Vietnamese-VTB/vi_vtb-ud-dev.conllu",
            "dat/dep/UD_Vietnamese-VTB/vi_vtb-ud-test.conllu",
            "dat/emb/cc.vi.300.vocab.vec",
            "dat/emb/numberbatch-vi-19.08.vocab.txt")
          case _ =>
            println("Invalid language code!")
            ("", "", "", "", "")
        }
        // read the training data set
        val df = readGraphs(spark, trainPath, config.maxSeqLen, config.las)
        // train a preprocessor and use it to transform training/dev data sets
        val preprocessor = createPipeline(df, config)
        val ef = preprocessor.transform(df)
        // read the validation data set
        val dfV = readGraphs(spark, validPath, config.maxSeqLen, config.las)
        val efV = preprocessor.transform(dfV)
        // read the test data set
        val dfW = readGraphs(spark, testPath, config.maxSeqLen, config.las)
        val efW = preprocessor.transform(dfW)
        println("#(trainGraphs) = " + df.count())
        println("#(validGraphs) = " + dfV.count())
        println("#(testGraphs) = " + dfW.count())
        // array of offset labels, index -> label: offsets[i] is the label at index i:
        val offsets = preprocessor.stages(0).asInstanceOf[CountVectorizerModel].vocabulary
        val offsetMap = offsets.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap // 1-based index for BigDL
        val offsetIndex = offsets.zipWithIndex.map(p => (p._2 + 1, p._1)).toMap
        val numOffsets = offsets.length
        val tokens = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
        val tokensMap = tokens.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap   // 1-based index for BigDL
        val numVocab = tokens.length
        val partsOfSpeech = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
        val partsOfSpeechMap = partsOfSpeech.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap // 1-based index for BigDL
        val numPartsOfSpeech = partsOfSpeech.length
        val featureStructures = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
        val featureStructureMap = featureStructures.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap // 1-based index for BigDL
        val numFeatureStructures = featureStructures.length
        println("   labels = " + offsets.mkString(", "))
        println("#(labels) = " + numOffsets)
        println(" #(vocab) = " + numVocab)
        println("   #(PoS) = " + numPartsOfSpeech)
        println("    #(fs) = " + numFeatureStructures)
        // extract token, pos and offset indices (start from 1 to use in BigDL)
        val tokenSequencer = new Sequencer(tokensMap, config.maxSeqLen, 0f).setInputCol("tokens").setOutputCol("t")
        val posSequencer = new Sequencer(partsOfSpeechMap, config.maxSeqLen, 0f).setInputCol("uPoS").setOutputCol("p")
        val featureSequencer = new Sequencer(featureStructureMap, config.maxSeqLen, 0f).setInputCol("featureStructure").setOutputCol("f")
        val offsetsSequencer = new Sequencer(offsetMap, config.maxSeqLen, -1f).setInputCol("offsets").setOutputCol("o")
        val gf = offsetsSequencer.transform(featureSequencer.transform(posSequencer.transform(tokenSequencer.transform(ef))))
        val gfV = offsetsSequencer.transform(featureSequencer.transform(posSequencer.transform(tokenSequencer.transform(efV))))
        val gfW = offsetsSequencer.transform(featureSequencer.transform(posSequencer.transform(tokenSequencer.transform(efW))))

        efV.select("tokens", "uPoS", "featureStructure").show(5)
        gfV.select("t", "p", "f", "o").show(5)

        // prepare train/valid/test data frame which are appropriate for each model type:
        val (uf, vf, wf) = config.modelType match {
          case "t" => (gf, gfV, gfW)
          case "tg" => (gf, gfV, gfW)
          case "tn" => (gf, gfV, gfW)
          case "t+p" | "tg+p" |  "tn+p" =>
            // assemble the two input vectors into one 
            val assembler = new VectorAssembler().setInputCols(Array("t", "p")).setOutputCol("t+p")
            val hf = assembler.transform(gf)
            val hfV = assembler.transform(gfV)
            val hfW = assembler.transform(gfW)
            (hf, hfV, hfW)
          case "b" =>
            val hf = gf.withColumn("b", createInputBERT(col("t")))
            val hfV = gfV.withColumn("b", createInputBERT(col("t")))
            val hfW = gfW.withColumn("b", createInputBERT(col("t")))
            (hf, hfV, hfW)
          case "f" => 
            // assemble the three input vectors into one
            val assembler = new VectorAssembler().setInputCols(Array("t", "p", "f")).setOutputCol("t+p+f")
            val hf = assembler.transform(gf)
            val hfV = assembler.transform(gfV)
            val hfW = assembler.transform(gfW)
            hfV.select("t+p+f", "o").show(5)
            (hf, hfV, hfW)
          case "x" =>
            val Array(hf, hfV, hfW) = addNodeEmbeddingFeatures(spark, config, Array(gf, gfV, gfW))
            (hf, hfV, hfW)
          case "bx" => 
            val gf2 = gf.withColumn("b", createInputBERT(col("t")))
            val gf2V = gfV.withColumn("b", createInputBERT(col("t")))
            val gf2W = gfW.withColumn("b", createInputBERT(col("t")))
            val Array(hf, hfV, hfW) = addNodeEmbeddingFeatures(spark, config, Array(gf2, gf2V, gf2W))
            (hf, hfV, hfW)
        }
        // create a BigDL model corresponding to a model type, return a tuple: (bigdl, featureSize, labelSize, featureColName)
        def createBigDL(config: ConfigDEP): (KerasNet[Float], Array[Array[Int]], Array[Int], String) = {
          config.modelType  match {      
            case "t" =>
              // 1. Sequential model with random token embeddings
              val bigdl = Sequential()
              bigdl.add(Embedding(numVocab + 1, config.tokenEmbeddingSize, inputLength = config.maxSeqLen).setName("tokEmbedding"))
              for (_ <- 1 to config.layers)
                bigdl.add(Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true)))
              bigdl.add(Dropout(config.dropoutRate))
              bigdl.add(Dense(numOffsets, activation = "softmax"))
              val (featureSize, labelSize) = (Array(Array(config.maxSeqLen)), Array(config.maxSeqLen))
              (bigdl, featureSize, labelSize, "t")
            case "tg" =>
              // 1.1 Sequential model with pretrained token embeddings (GloVe)
              val bigdl = Sequential()
              bigdl.add(WordEmbeddingP(gloveFile, tokensMap, inputLength = config.maxSeqLen))
              for (_ <- 1 to config.layers)
                bigdl.add(Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true)))
              bigdl.add(Dropout(config.dropoutRate))
              bigdl.add(Dense(numOffsets, activation = "softmax"))
              val (featureSize, labelSize) = (Array(Array(config.maxSeqLen)), Array(config.maxSeqLen))
              (bigdl, featureSize, labelSize, "t")
            case "tn" =>
              // 1.2 Sequential model with pretrained token embeddings (Numberbatch of ConceptNet)
              val bigdl = Sequential()
              bigdl.add(WordEmbeddingP(numberbatchFile, tokensMap, inputLength = config.maxSeqLen))
              for (_ <- 1 to config.layers)
                bigdl.add(Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true)))
              bigdl.add(Dropout(config.dropoutRate))
              bigdl.add(Dense(numOffsets, activation = "softmax"))
              val (featureSize, labelSize) = (Array(Array(config.maxSeqLen)), Array(config.maxSeqLen))
              (bigdl, featureSize, labelSize, "t")
            case "t+p"  | "tg+p" |  "tn+p" =>
              // A model for (token ++ partsOfSpeech) tensor
              val input = Input(inputShape = Shape(2*config.maxSeqLen), name = "input")
              val reshape = Reshape(targetShape = Array(2, config.maxSeqLen)).setName("reshape").inputs(input)
              val split = SplitTensor(1, 2).inputs(reshape)
              val token = SelectTable(0).setName("inputT").inputs(split)
              val inputT = Squeeze(1).inputs(token)
              val partsOfSpeech = SelectTable(1).setName("inputP").inputs(split)
              val inputP = Squeeze(1).inputs(partsOfSpeech)
              val embeddingT = config.modelType match {
                case "t+p" => Embedding(numVocab + 1, config.tokenEmbeddingSize).setName("tokEmbedding").inputs(inputT)
                case "tg+p" => WordEmbeddingP(gloveFile, tokensMap, inputLength = config.maxSeqLen).setName("tokenEmbedding").inputs(inputT)
                case "tn+p" => WordEmbeddingP(numberbatchFile, tokensMap, inputLength = config.maxSeqLen).setName("tokenEmbedding").inputs(inputT)
              }
              val embeddingP = Embedding(numPartsOfSpeech + 1, config.partsOfSpeechEmbeddingSize).setName("posEmbedding").inputs(inputP)
              val merge = Merge.merge(inputs = List(embeddingT, embeddingP), mode = "concat")
              val rnn1 = Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true).setName("LSTM-1")).inputs(merge)
              val rnn2 = Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true).setName("LSTM-2")).inputs(rnn1)
              val dropout = Dropout(config.dropoutRate).inputs(rnn2)
              val output = Dense(numOffsets, activation = "softmax").setName("dense").inputs(dropout)
              val bigdl = Model(input, output)
              val (featureSize, labelSize) = (Array(Array(2*config.maxSeqLen)), Array(config.maxSeqLen))
              (bigdl, featureSize, labelSize, "t+p")
            case "b" =>
              // 4. BERT model using one input of 4*maxSeqLen elements
              val input = Input(inputShape = Shape(4*config.maxSeqLen), name = "input")
              val reshape = Reshape(targetShape = Array(4, config.maxSeqLen)).inputs(input)
              val split = SplitTensor(1, 4).inputs(reshape)
              val selectIds = SelectTable(0).setName("inputId").inputs(split)
              val inputIds = Squeeze(1).inputs(selectIds)
              val selectSegments = SelectTable(1).setName("segmentId").inputs(split)
              val segmentIds = Squeeze(1).inputs(selectSegments)
              val selectPositions = SelectTable(2).setName("positionId").inputs(split)
              val positionIds = Squeeze(1).inputs(selectPositions)
              val selectMasks = SelectTable(3).setName("masks").inputs(split)
              val masksReshaped = Reshape(targetShape = Array(1, 1, config.maxSeqLen)).setName("mask").inputs(selectMasks)
              val bert = BERT(vocab = numVocab + 1, hiddenSize = config.tokenEmbeddingSize, nBlock = config.layers, nHead = config.heads, maxPositionLen = config.maxSeqLen,
                intermediateSize = config.tokenHiddenSize, outputAllBlock = false).setName("bert")
              val bertNode = bert.inputs(Array(inputIds, segmentIds, positionIds, masksReshaped))
              val bertOutput = SelectTable(0).setName("firstBlock").inputs(bertNode)
              val dense = Dense(numOffsets).setName("dense").inputs(bertOutput)
              val output = SoftMax().setName("output").inputs(dense)
              val bigdl = Model(input, output)
              val (featureSize, labelSize) = (Array(Array(4*config.maxSeqLen)), Array(config.maxSeqLen))
              (bigdl, featureSize, labelSize, "b")
            case "f" => 
              // A model for (token ++ uPoS ++ featureStructure) tensor
              val inputT = Input(inputShape = Shape(config.maxSeqLen), name = "inputT") 
              val inputP = Input(inputShape = Shape(config.maxSeqLen), name = "inputP") 
              val inputF = Input(inputShape = Shape(config.maxSeqLen), name = "inputF") 
              val embeddingT = Embedding(numVocab + 1, config.tokenEmbeddingSize).setName("tokEmbedding").inputs(inputT)
              val embeddingP = Embedding(numPartsOfSpeech + 1, config.partsOfSpeechEmbeddingSize).setName("posEmbedding").inputs(inputP)
              val embeddingF = Embedding(numFeatureStructures + 1, config.featureStructureEmbeddingSize).setName("fsEmbedding").inputs(inputF)

              val merge = Merge.merge(inputs = List(embeddingT, embeddingP, embeddingF), mode = "concat")
              val rnn1 = Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true).setName("LSTM-1")).inputs(merge)
              val rnn2 = Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true).setName("LSTM-2")).inputs(rnn1)
              val dropout = Dropout(config.dropoutRate).inputs(rnn2)
              val output = Dense(numOffsets, activation = "softmax").setName("dense").inputs(dropout)
              val bigdl = Model(Array(inputT, inputP, inputF), output) // multiple inputs
              val (featureSize, labelSize) = (Array(Array(config.maxSeqLen), Array(config.maxSeqLen), Array(config.maxSeqLen)), Array(config.maxSeqLen))
              (bigdl, featureSize, labelSize, "t+p+f") 
            case "x" =>
              // A model for (token ++ uPoS ++ featureStructure ++ graphX ++ Node2Vec) tensor
              val inputT = Input(inputShape = Shape(config.maxSeqLen), name = "inputT") 
              val inputP = Input(inputShape = Shape(config.maxSeqLen), name = "inputP") 
              val inputF = Input(inputShape = Shape(config.maxSeqLen), name = "inputF") 
              val embeddingT = Embedding(numVocab + 1, config.tokenEmbeddingSize).setName("tokEmbedding").inputs(inputT)
              val embeddingP = Embedding(numPartsOfSpeech + 1, config.partsOfSpeechEmbeddingSize).setName("posEmbedding").inputs(inputP)
              val embeddingF = Embedding(numFeatureStructures + 1, config.featureStructureEmbeddingSize).setName("fsEmbedding").inputs(inputF)
              // graphX input
              val inputX1 = Input(inputShape = Shape(3*config.maxSeqLen), name = "inputX1") // 3 for graphX
              val reshapeX1 = Reshape(targetShape = Array(config.maxSeqLen, 3)).setName("reshapeX1").inputs(inputX1)
              // Node2Vec input
              val inputX2 = Input(inputShape = Shape(32*config.maxSeqLen), name = "inputX2") // 32 for node2vec
              val reshapeX2 = Reshape(targetShape = Array(config.maxSeqLen, 32)).setName("reshapeX2").inputs(inputX2)

              val merge = Merge.merge(inputs = List(embeddingT, embeddingP, embeddingF, reshapeX1, reshapeX2), mode = "concat")
              val rnn1 = Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true).setName("LSTM-1")).inputs(merge)
              val rnn2 = Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true).setName("LSTM-2")).inputs(rnn1)
              val dropout = Dropout(config.dropoutRate).inputs(rnn2)
              val output = Dense(numOffsets, activation = "softmax").setName("dense").inputs(dropout)
              val bigdl = Model(Array(inputT, inputP, inputF, inputX1, inputX2), output) // multiple inputs
              val (featureSize, labelSize) = (
                Array(Array(config.maxSeqLen), Array(config.maxSeqLen), Array(config.maxSeqLen), Array(3*config.maxSeqLen), Array(32*config.maxSeqLen)), 
                Array(config.maxSeqLen))
              (bigdl, featureSize, labelSize, "t+p+f+x")
            case "bx" => 
              // A model for (token ++ graphX ++ Node2Vec) tensor using BERT 
              // BERT model using one input of 4*maxSeqLen elements
              val input = Input(inputShape = Shape(4*config.maxSeqLen), name = "input")
              val reshape = Reshape(targetShape = Array(4, config.maxSeqLen)).inputs(input)
              val split = SplitTensor(1, 4).inputs(reshape)
              val selectIds = SelectTable(0).setName("inputId").inputs(split)
              val inputIds = Squeeze(1).inputs(selectIds)
              val selectSegments = SelectTable(1).setName("segmentId").inputs(split)
              val segmentIds = Squeeze(1).inputs(selectSegments)
              val selectPositions = SelectTable(2).setName("positionId").inputs(split)
              val positionIds = Squeeze(1).inputs(selectPositions)
              val selectMasks = SelectTable(3).setName("masks").inputs(split)
              val masksReshaped = Reshape(targetShape = Array(1, 1, config.maxSeqLen)).setName("mask").inputs(selectMasks)
              val bert = BERT(vocab = numVocab + 1, hiddenSize = config.tokenEmbeddingSize, nBlock = config.layers, nHead = config.heads, maxPositionLen = config.maxSeqLen,
                intermediateSize = config.tokenHiddenSize, outputAllBlock = false).setName("bert")
              val bertNode = bert.inputs(Array(inputIds, segmentIds, positionIds, masksReshaped))
              val bertOutput = SelectTable(0).setName("firstBlock").inputs(bertNode)
              // graphX input
              val inputX1 = Input(inputShape = Shape(3*config.maxSeqLen), name = "inputX1") // 3 for graphX
              val reshapeX1 = Reshape(targetShape = Array(config.maxSeqLen, 3)).setName("reshapeX1").inputs(inputX1)
              // Node2Vec input
              val inputX2 = Input(inputShape = Shape(32*config.maxSeqLen), name = "inputX2") // 32 for node2vec
              val reshapeX2 = Reshape(targetShape = Array(config.maxSeqLen, 32)).setName("reshapeX2").inputs(inputX2)
              // merge bert ++ graphX ++ Node2Vec
              val merge = Merge.merge(inputs = List(bertOutput, reshapeX1, reshapeX2), mode = "concat")
              val dense = Dense(numOffsets).setName("dense").inputs(merge)
              val output = SoftMax().setName("output").inputs(dense)
              val bigdl = Model(Array(input, inputX1, inputX2), output)
              val (featureSize, labelSize) = (Array(Array(4*config.maxSeqLen), Array(3*config.maxSeqLen), Array(32*config.maxSeqLen)), Array(config.maxSeqLen))
              (bigdl, featureSize, labelSize, "bx")
          }
        }
        
        val (bigdl, featureSize, labelSize, featureColName) = createBigDL(config)

        def weights(): Tensor[Float] = {
          // compute label weights for the loss function
          import spark.implicits._
          val xf = df.select("offsets").flatMap(row => row.getSeq[String](0))
          val yf = xf.groupBy("value").count()
          val labelFreq = yf.select("value", "count").collect().map(row => (offsetMap(row.getString(0)), row.getLong(1)))
          val total = labelFreq.map(_._2).sum.toFloat
          val ws = labelFreq.map(p => (p._1, p._2 / total)).toMap
          val tensor = Tensor(ws.size)
          for (j <- ws.keys)
            tensor(j) = 1f/ws(j) // give higher weight to minority labels
          tensor
        }
        val numCores = Runtime.getRuntime().availableProcessors()
        val batchSize = if (config.batchSize % numCores != 0) numCores * 4; else config.batchSize
        val modelPath = s"${config.modelPath}/${config.language}-${config.modelType}.bigdl"

        def sanityCheckTokenEmbedding(bigdl: KerasNet[Float]): Unit = {
          val weightsBias = bigdl.getSubModules().filter(p => p.getName() == "tokEmbedding").head.getWeightsBias().head
          val Array(m, n) = weightsBias.size()
          val weightsOfFirstRow = (1 to n).map(j => weightsBias(Array(1, j)))
          println(s"tokEmbedding layer size = $m x $n, sum of first row = ${weightsOfFirstRow.sum}")
        }

        // eval() needs an array of feature column names for a proper reshaping of input tensors
        val featureColNames = config.modelType match {
          case "f" => Array("t", "p", "f")
          case "x" => Array("t", "p", "f", "x1", "x2")
          case "bx" => Array("b", "x1", "x2")
          case _ => Array(featureColName)
        }

        // best maxIterations for each language which is validated on the dev. split:
        val maxIterations = config.language match {
          case "eng" => 2000
          case "fra" => 2000
          case "ind" => 800
          case "vie" => 600
          case _ => 1000
        }  
    
        // create a training criterion, it is necessary to set sizeAverage of ClassNLLCriterion to false in non-batch mode            
        val criterion = if (config.weightedLoss) {
          TimeDistributedMaskCriterion(ClassNLLCriterion(weights = weights(), sizeAverage = false, logProbAsInput = false, paddingValue = -1), paddingValue = -1)
        } else {
          TimeDistributedMaskCriterion(ClassNLLCriterion(sizeAverage = false, logProbAsInput = false, paddingValue = -1), paddingValue = -1)
        }
        config.mode match {
          case "train" =>
            println("config = " + config)
            val estimator = NNEstimator(bigdl, criterion, featureSize, labelSize)
            val trainingSummary = TrainSummary(appName = config.modelType, logDir = s"sum/dep/${config.language}")
            val validationSummary = ValidationSummary(appName = config.modelType, logDir = s"sum/dep/${config.language}")
            estimator.setLabelCol("o").setFeaturesCol(featureColName)
              .setBatchSize(batchSize)
              .setOptimMethod(new Adam(config.learningRate))
              .setTrainSummary(trainingSummary)
              .setValidationSummary(validationSummary)
              .setValidation(Trigger.everyEpoch, vf, Array(new TimeDistributedTop1Accuracy(-1), new Loss[Float](criterion)), batchSize)
              .setEndWhen(Trigger.or(Trigger.maxEpoch(config.epochs), Trigger.maxIteration(maxIterations)))
            // train
            estimator.fit(uf)
            // save the model
            bigdl.saveModel(modelPath, overWrite = true)
            // evaluate the model on the training/dev sets
            // sanityCheckTokenEmbedding(bigdl)
            val scores = eval(bigdl, config, uf, vf, featureColNames)
            val heads = if (config.modelType != "b") 0 else config.heads
            val result = f"\n${config.language};${config.modelType};${config.tokenEmbeddingSize};${config.tokenHiddenSize};${config.layers};$heads;${scores(0)}%.4g;${scores(1)}%.4g"
            println(result)
            val suffix = if (config.las) "las" else "uas"
            Files.write(Paths.get(s"${config.scorePath}-$suffix.tsv"), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
          case "eval" =>
            // load the bigdl model
            println(s"Loading model in the path: $modelPath...")
            val bigdl = Models.loadModel[Float](modelPath)
            bigdl.summary()
            // sanity check of some parameters 
            // sanityCheckTokenEmbedding(bigdl)
            
            // write out training/dev scores
            val scores = eval(bigdl, config, uf, vf, featureColNames)

            val heads = if (config.modelType != "b") 0 else config.heads
            val result = f"\n${config.language};${config.modelType};${config.tokenEmbeddingSize};${config.tokenHiddenSize};${config.layers};$heads;${scores(0)}%.4g;${scores(1)}%.4g"
            println(result)
            val suffix = if (config.las) "las" else "uas"
            Files.write(Paths.get(s"${config.scorePath}-$suffix.tsv"), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
          case "validate" => 
            // perform a series of experiments to find the best hyper-params on the development set for a language
            // The arguments are: -l <lang> -t <modelType> -m validate
            val ws = Array(64, 128, 200)
            val hs = Array(64, 128, 200, 300)
            for (_ <- 1 to 3) {
              for (w <- ws; h <- hs) {
                val cfg = config.copy(tokenEmbeddingSize = w, tokenHiddenSize = h)
                println(cfg)
                val (bigdl, featureSize, labelSize, featureColName) = createBigDL(cfg)
                val estimator = NNEstimator(bigdl, criterion, featureSize, labelSize)
                estimator.setLabelCol("o").setFeaturesCol(featureColName)
                  .setBatchSize(batchSize)
                  .setOptimMethod(new Adam(config.learningRate))
                  .setEndWhen(Trigger.or(Trigger.maxEpoch(config.epochs), Trigger.maxIteration(maxIterations)))
                // train
                estimator.fit(uf)
                val scores = eval(bigdl, cfg, uf, vf, featureColNames)
                val result = f"\n${cfg.language};${cfg.modelType};$w;$h;2;0;${scores(0)}%.4g;${scores(1)}%.4g"
                val suffix = if (config.las) "las" else "uas"
                Files.write(Paths.get(s"${config.scorePath}-$suffix.tsv"), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
              }
            }
          case "validate-b" => 
            // perform a series of experiments to find the best hyper-params on the development set for a language
            // The arguments are: -l <lang> -t b/bx -m validate
            val ws = Array(64, 128, 200)
            val hs = Array(64, 128, 200, 300)
            val js = Array(2, 3)
            val nHeads = Array(2, 4, 8)
            for (_ <- 1 to 3) {
              for (w <- ws; h <- hs; j <- js; n <- nHeads) {
                val cfg = config.copy(tokenEmbeddingSize = w, tokenHiddenSize = h, layers = j, heads = n)
                println(cfg)
                val (bigdl, featureSize, labelSize, featureColName) = createBigDL(cfg)
                val estimator = NNEstimator(bigdl, criterion, featureSize, labelSize)
                estimator.setLabelCol("o").setFeaturesCol(featureColName)
                  .setBatchSize(batchSize)
                  .setOptimMethod(new Adam(config.learningRate))
                  .setEndWhen(Trigger.or(Trigger.maxEpoch(config.epochs), Trigger.maxIteration(maxIterations)))
                // train
                estimator.fit(uf)
                val scores = eval(bigdl, cfg, uf, vf, featureColNames)
                val result = f"\n${cfg.language};${cfg.modelType};$w;$h;$j;$n;${scores(0)}%.4g;${scores(1)}%.4g"
                val suffix = if (config.las) "las" else "uas"
                Files.write(Paths.get(s"${config.scorePath}-$suffix.tsv"), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
              }
            }
          case "predict" =>
            // train the model on the training set (uf) using the best hyper-parameters which was tuned on the validation set (vf)
            // and run prediction it on the test set (wf) to collect scores
            val estimator = NNEstimator(bigdl, criterion, featureSize, labelSize)
            val trainingSummary = TrainSummary(appName = config.modelType, logDir = s"sum/dep/${config.language}")
            val validationSummary = ValidationSummary(appName = config.modelType, logDir = s"sum/dep/${config.language}")
            estimator.setLabelCol("o").setFeaturesCol(featureColName)
              .setBatchSize(batchSize)
              .setOptimMethod(new Adam(config.learningRate))
              .setTrainSummary(trainingSummary)
              .setValidationSummary(validationSummary)
              .setValidation(Trigger.everyEpoch, vf, Array(new TimeDistributedTop1Accuracy(-1), new Loss[Float](criterion)), batchSize)
              .setEndWhen(Trigger.or(Trigger.maxEpoch(config.epochs), Trigger.maxIteration(maxIterations)))
            // train
            estimator.fit(uf)
            // save the model
            bigdl.saveModel(modelPath, overWrite = true)
            // evaluate the model on the training and test set
            val scores = eval(bigdl, config, uf, wf, featureColNames)
            val heads = if (config.modelType != "b") 0 else config.heads
            val result = f"\n${config.language};${config.modelType};${config.tokenEmbeddingSize};${config.tokenHiddenSize};${config.layers};$heads;${scores(0)}%.4g;${scores(1)}%.4g"
            println(result)
            val suffix = if (config.las) "las" else "uas"
            Files.write(Paths.get(s"${config.scorePath}-$suffix.tsv"), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
          case "preprocess" =>
            val af = df.union(dfV).union(dfW)
            val preprocessor = createPipeline(af, config)
            val vocab = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary.toSet
            println("#(vocab) = " + vocab.size)
            println(vocab)
          case "statistic" => 
            val graphs = GraphReader.read(trainPath)
            val lengths = graphs.map(g => g.sentence.tokens.size)
            val gs = lengths.groupBy(identity).map { case (k, v) => (k, v.size) }.toList
            val hs = gs.sortBy(_._1)
            hs.foreach(println)
        }
        spark.stop()
      case None => println("Invalid config!")
    }
  }
}
