package vlp.dep

import scopt.OptionParser
import org.apache.spark.SparkConf
import com.intel.analytics.bigdl.dllib.NNContext
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.ml._
import org.apache.spark.ml.feature._

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.keras.models.{Model, Models, KerasNet}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.{ClassNLLCriterion, TimeDistributedMaskCriterion}
import com.intel.analytics.bigdl.phuonglh.JointClassNLLCriterion
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.{Trigger, Loss}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}

import vlp.ner.{ArgMaxLayer, Sequencer, TimeDistributedTop1Accuracy}
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import com.intel.analytics.bigdl.dllib.nn.Transpose
import org.apache.spark.sql.functions._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import com.intel.analytics.bigdl.dllib.nn.{Sequential => NSequential}
import com.intel.analytics.bigdl.dllib.nn.JoinTable
import com.intel.analytics.bigdl.dllib.nn.MapTable



/**
  * Label-Offset Prediction Dependency Parser.
  * <p/>
  * Phuong LE-HONG, phuonglh@gmail.com
  * 
  */
object LOP {

  def readGraphs(spark: SparkSession, config: ConfigDEP): (DataFrame, DataFrame, DataFrame) = {
    val (trainPath, validPath, testPath) = config.language match {
      case "eng" => ("dat/dep/UD_English-EWT/en_ewt-ud-train.conllu",
        "dat/dep/UD_English-EWT/en_ewt-ud-dev.conllu",
        "dat/dep/UD_English-EWT/en_ewt-ud-test.conllu")
      case "fra" => ("dat/dep/UD_French-GSD/fr_gsd-ud-train.conllu",
        "dat/dep/UD_French-GSD/fr_gsd-ud-dev.conllu",
        "dat/dep/UD_French-GSD/fr_gsd-ud-test.conllu")
      case "ind" => ("dat/dep/UD_Indonesian-GSD/id_gsd-ud-train.conllu",
        "dat/dep/UD_Indonesian-GSD/id_gsd-ud-dev.conllu",
        "dat/dep/UD_Indonesian-GSD/id_gsd-ud-test.conllu")
      case "vie" => ("dat/dep/UD_Vietnamese-VTB/vi_vtb-ud-train.conllu",
        "dat/dep/UD_Vietnamese-VTB/vi_vtb-ud-dev.conllu",
        "dat/dep/UD_Vietnamese-VTB/vi_vtb-ud-test.conllu")
      case _ =>
        println("Invalid language code!")
        ("", "", "")
    }
    (
      DEPx.readGraphs(spark, trainPath, config.maxSeqLen, config.las), 
      DEPx.readGraphs(spark, validPath, config.maxSeqLen, config.las), 
      DEPx.readGraphs(spark, testPath, config.maxSeqLen, config.las)
    )
  }

  def createPipeline(df: DataFrame, config: ConfigDEP): PipelineModel = {
    val vectorizerOffsets = new CountVectorizer().setInputCol("offsets").setOutputCol("off")
    val vectorizerLabels = new CountVectorizer().setInputCol("labels").setOutputCol("dep")
    val vectorizerTokens = new CountVectorizer().setInputCol("tokens").setOutputCol("tok").setVocabSize(config.maxVocabSize)
    val vectorizerPartsOfSpeech = new CountVectorizer().setInputCol("uPoS").setOutputCol("pos")
    val vectorizerFeatureStructure = new CountVectorizer().setInputCol("featureStructure").setOutputCol("fst")
    val stages = Array(vectorizerOffsets, vectorizerLabels, vectorizerTokens, vectorizerPartsOfSpeech, vectorizerFeatureStructure)
    val pipeline = new Pipeline().setStages(stages)
    pipeline.fit(df)
  }

  /**
    * create a BigDL model corresponding to a model type, return 
    *
    * @param config configuration
    * @param numVocab number of input tokens
    * @param numOffsets number of offsets
    * @param numLabels number of dependency labels
    * @return a tuple: (bigdl, featureSize, labelSize, featureColName) 
  */
  def createBigDL(config: ConfigDEP, numVocab: Int, numOffsets: Int, numLabels: Int): (KerasNet[Float], Array[Array[Int]], Array[Int], String) = {
    config.modelType  match {      
      case "t" =>
        val inputT = Input(inputShape = Shape(config.maxSeqLen), name = "inputT")
        val embeddingT = Embedding(numVocab + 1, config.tokenEmbeddingSize).setName("tokEmbedding").inputs(inputT)
        val offsetRNN1 = Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true).setName("LSTM-o1")).inputs(embeddingT)
        val offsetRNN2 = Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true).setName("LSTM-o2")).inputs(offsetRNN1)
        val labelRNN1 = Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true).setName("LSTM-d1")).inputs(embeddingT)
        val labelRNN2 = Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true).setName("LSTM-d2")).inputs(labelRNN1)
        val offsetOutput = Dense(numOffsets, activation = "softmax").setName("denseO").inputs(offsetRNN2)
        val labelOutput =  Dense(numLabels, activation = "softmax").setName("denseD").inputs(labelRNN2)
        // transpose in order to apply zeroPadding
        val transposeOffset = new KerasLayerWrapper(new Transpose[Float](permutations = Array((2, 3)))).inputs(offsetOutput)
        val transposeLabel = new KerasLayerWrapper(new Transpose[Float](permutations = Array((2, 3)))).inputs(labelOutput)
        // right-pad the smaller size
        val padSize = Math.abs(numOffsets - numLabels)
        val zeroPadding = if (numOffsets < numLabels) {
            new ZeroPadding1D(padding = Array(0, padSize)).inputs(transposeOffset) 
        } else {
            new ZeroPadding1D(padding = Array(0, padSize)).inputs(transposeLabel)
        }
        val transposeBack = new KerasLayerWrapper(new Transpose[Float](permutations = Array((2, 3)))).inputs(zeroPadding)
        val merge = if (numOffsets < numLabels) {
          Merge.merge(inputs = List(transposeBack, labelOutput), mode = "concat", concatAxis = -1)
        } else {
          Merge.merge(inputs = List(offsetOutput, transposeBack), mode = "concat", concatAxis = -1)
        }
        // output size is maxSeqLen x (2*max(numOffsets, numLabels))
        // duplicate to have the same seqLen as labels (for TimeDistributedCriterion to work)
        val duplicate = Merge.merge(inputs = List(merge, merge), mode = "concat", concatAxis = 1)
        val bigdl = Model(inputT, duplicate)
        val (featureSize, labelSize) = (Array(Array(config.maxSeqLen)), Array(2*config.maxSeqLen))
        (bigdl, featureSize, labelSize, "t")
      case "b" => 
        val bigdl = Sequential()
        val (featureSize, labelSize) = (Array(Array(4*config.maxSeqLen)), Array(2*config.maxSeqLen))
        (bigdl, featureSize, labelSize, "t")
    }
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
    // the output of bigdl contains duplicates (2 similar halves). First, select only the first half
    val half = SplitTensor(1, 2)
    val select = SelectTable(0)
    sequential.add(half).add(select)
    // split the tensor into 2 halves: (20 x 74) => (20 x 37) and (20 x 37)
    val transpose = new KerasLayerWrapper(new Transpose[Float](permutations = Array((2, 3))))
    val split = SplitTensor(1, 2)
    val transposeBack = new Transpose[Float](permutations = Array((2, 3)))
    // pass the table of 2 elements into a map table, each entry has the same sequential module
    val module = NSequential[Float]().add(transposeBack).add(vlp.ner.ArgMax())
    val parallelTable = new KerasLayerWrapper(MapTable[Float]().add(module))
    val joinTable = new KerasLayerWrapper(JoinTable[Float](2, 3))
    sequential.add(transpose).add(split).add(parallelTable).add(joinTable)
    sequential.summary()

    // run prediction on the training set and validation set
    val predictions = Array(
      sequential.predict(uf, featureCols = featureColNames, predictionCol = "z"),
      sequential.predict(vf, featureCols = featureColNames, predictionCol = "z")
    )
    val spark = SparkSession.getActiveSession.get
    import spark.implicits._
    val scores = for (prediction <- predictions) yield {
      val zf = prediction.select("o+d", "z").map { row =>
        val o = row.getAs[linalg.Vector](0).toArray
        val p = row.getSeq[Float](1)
        // split o and p into two halves, zip the correct/prediction arrays, remove padding values and compute matches
        val offsetMatch = o.take(o.size/2).zip(p.take(o.size/2)).filter(p => p._1 > 0).map(p => p._1 == p._2)
        val labelMatch =  o.takeRight(p.size/2).zip(p.takeRight(p.size/2)).filter(p => p._1 > 0).map(p => p._1 == p._2)
        val correct = offsetMatch.zip(labelMatch).map(p => if (p._1 && p._2) 1 else 0).sum
        val total = offsetMatch.size
        (correct, total)
      }
      // show the result
      val af = zf.toDF("correct", "total")
      af.show()
      val agg = af.agg(sum("correct"), sum("total")).first
      val c = agg.getLong(0)
      val t = agg.getLong(1)
      c.toDouble/t
    }
    scores
  }

  def main(args: Array[String]): Unit = {
    DEPx.parseOptions.parse(args, ConfigDEP()) match {
      case Some(config) =>
        val conf = new SparkConf().setAppName(getClass.getName).setMaster(config.master)
          .set("spark.driver.memory", config.driverMemory)
          .set("spark.executor.memory", config.executorMemory)
        val sc = NNContext.initNNContext(conf)
        sc.setLogLevel("ERROR")
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
        // read the data sets
        val (df, dfV, dfW) = readGraphs(spark, config)
        println("#(trainGraphs) = " + df.count())
        println("#(validGraphs) = " + dfV.count())
        println("#(testGraphs) = " + dfW.count())
        // create a preprocessing pipeline and train it on the training set, then transform three sets
        val preprocessor = createPipeline(df, config)
        val (ef, efV, efW) = (preprocessor.transform(df), preprocessor.transform(dfV), preprocessor.transform(dfW))
        // extract vocabs and create dictionaries for offset, labels, tokens...
        val offsets = preprocessor.stages(0).asInstanceOf[CountVectorizerModel].vocabulary
        val offsetMap = offsets.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
        val offsetIndex = offsets.zipWithIndex.map(p => (p._2 + 1, p._1)).toMap
        val numOffsets = offsets.length
        val dependencies = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
        val dependencyMap = dependencies.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
        val dependencyIndex = dependencies.zipWithIndex.map(p => (p._2 + 1, p._1)).toMap
        val numDependencies = dependencies.length
        val tokens = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
        val tokensMap = tokens.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
        val numVocab = tokens.length
        val partsOfSpeech = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
        val partsOfSpeechMap = partsOfSpeech.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
        val numPartsOfSpeech = partsOfSpeech.length
        println("   offsets = " + offsets.mkString(", "))
        println("#(offsets) = " + numOffsets)
        println("    labels = " + dependencies.mkString(", "))
        println(" #(labels) = " + numDependencies)
        println("  #(vocab) = " + numVocab)
        println("    #(PoS) = " + numPartsOfSpeech)
        // 
        // sequencer tokens, pos, offsets and labels; then concatenate offsets and labels
        val tokenSequencer = new Sequencer(tokensMap, config.maxSeqLen, 0f).setInputCol("tokens").setOutputCol("t")
        val posSequencer = new Sequencer(partsOfSpeechMap, config.maxSeqLen, 0f).setInputCol("uPoS").setOutputCol("p")
        val offsetSequencer = new Sequencer(offsetMap, config.maxSeqLen, -1f).setInputCol("offsets").setOutputCol("o")
        val labelSequencer = new Sequencer(dependencyMap, config.maxSeqLen, -1f).setInputCol("labels").setOutputCol("d")
        val outputAssembler = new VectorAssembler().setInputCols(Array("o", "d")).setOutputCol("o+d")
        val labelShifter = new Shifter(numOffsets, -1f).setInputCol("d").setOutputCol("d1")
        val outputAssembler1 = new VectorAssembler().setInputCols(Array("o", "d1")).setOutputCol("o+d1")
        val pipeline = new Pipeline().setStages(Array(tokenSequencer, posSequencer, offsetSequencer, 
          labelSequencer, labelShifter, outputAssembler, outputAssembler1))
        val pipelineModel = pipeline.fit(df)

        val (ff, ffV, ffW) = (pipelineModel.transform(ef), pipelineModel.transform(efV), pipelineModel.transform(efW))
        ff.select("o+d", "o+d1").show(5, false)

        val (uf, vf, wf) = config.modelType match {
          case "t" => (ff, ffV, ffW)
        }
        val (bigdl, featureSize, labelSize, featureColName) = createBigDL(config, numVocab, numOffsets, numDependencies)
        val numCores = Runtime.getRuntime().availableProcessors()
        val batchSize = if (config.batchSize % numCores != 0) numCores * 4; else config.batchSize
        val modelPath = s"${config.modelPath}/${config.language}-${config.modelType}.bigdl"
        val loss = ClassNLLCriterion(sizeAverage = false, logProbAsInput = false, paddingValue = -1)
        val criterion = TimeDistributedMaskCriterion(loss, paddingValue = -1)
        // eval() needs an array of feature column names for a proper reshaping of input tensors
        val featureColNames = config.modelType match {
          case "x" => Array("t", "p", "x", "x2") // TODO
          case "bx" => Array("tb", "x", "x2") // TODO
          case _ => Array(featureColName)
        }
        // best maxIterations for each language which is validated on the dev. split:
        val maxIterations = config.language match {
          case "eng" => 2000
          case "fra" => 2000
          case "ind" => 1000
          case "vie" => 1000
          case _ => 1000
        }
        config.mode match {
          case "train" =>
            println("config = " + config)
            bigdl.summary()
            val estimator = NNEstimator(bigdl, criterion, featureSize, labelSize)
            val trainingSummary = TrainSummary(appName = config.modelType, logDir = s"sum/lop/${config.language}")
            val validationSummary = ValidationSummary(appName = config.modelType, logDir = s"sum/lop/${config.language}")
            estimator.setLabelCol("o+d1").setFeaturesCol(featureColName)
              .setBatchSize(batchSize) // global batch size
              .setOptimMethod(new Adam(config.learningRate))
              .setTrainSummary(trainingSummary)
              .setValidationSummary(validationSummary)
              .setValidation(Trigger.everyEpoch, vf, Array(new TimeDistributedTop1Accuracy(-1), new Loss[Float](criterion)), batchSize) 
              .setEndWhen(Trigger.or(Trigger.maxEpoch(config.epochs), Trigger.maxIteration(maxIterations)))
            estimator.fit(uf)
            bigdl.saveModel(modelPath, overWrite = true)
            // evaluate the model on the training/dev sets
            val scores = eval(bigdl, config, uf, vf, featureColNames)
            val heads = if (config.modelType != "b") 0 else config.heads
            val result = f"\n${config.language};${config.modelType};${config.tokenEmbeddingSize};${config.tokenHiddenSize};${config.layers};$heads;${scores(0)}%.4g;${scores(1)}%.4g"
            println(result)
            Files.write(Paths.get(s"${config.scorePath}-las.tsv"), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
          case "eval" =>
            println(s"Loading model in the path: $modelPath...")
            val bigdl = Models.loadModel[Float](modelPath)
            bigdl.summary()
            val scores = eval(bigdl, config, uf, vf, Array(featureColName))

            val heads = if (config.modelType != "b") 0 else config.heads
            val result = f"\n${config.language};${config.modelType};${config.tokenEmbeddingSize};${config.tokenHiddenSize};${config.layers};$heads;${scores(0)}%.4g;${scores(1)}%.4g"
            println(result)
            Files.write(Paths.get(s"${config.scorePath}-las.tsv"), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
          case "validate" => 
            // perform a series of experiments to find the best hyper-params on the development set for a language
            // The arguments are: -l <lang> -t <modelType> -m validate
            val ws = Array(128, 200)
            val hs = Array(128, 200, 300, 400)
            for (_ <- 1 to 3) {
              for (w <- ws; h <- hs) {
                val cfg = config.copy(tokenEmbeddingSize = w, tokenHiddenSize = h)
                println(cfg)
                val (bigdl, featureSize, labelSize, featureColName) = createBigDL(cfg, numVocab, numOffsets, numDependencies)
                val estimator = NNEstimator(bigdl, criterion, featureSize, labelSize)
                estimator.setLabelCol("o+d1").setFeaturesCol(featureColName)
                  .setBatchSize(batchSize)
                  .setOptimMethod(new Adam(config.learningRate))
                  .setEndWhen(Trigger.or(Trigger.maxEpoch(config.epochs), Trigger.maxIteration(maxIterations)))
                // train
                estimator.fit(uf)
                val scores = eval(bigdl, cfg, uf, vf, featureColNames)
                val result = f"\n${cfg.language};${cfg.modelType};$w;$h;2;0;${scores(0)}%.4g;${scores(1)}%.4g"
                Files.write(Paths.get(s"${config.scorePath}-las.tsv"), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
              }
            }
          }
      case None => {}
    }
  }
}
