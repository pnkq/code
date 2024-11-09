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
import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.apache.spark.sql.expressions.Window



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
    * @param numPartsOfSpeech number of parts of speech
    * @param numFeatureStructures number of feature structures
    * @return a tuple: (bigdl, featureSize, labelSize, featureColName) 
  */
  def createBigDL(config: ConfigDEP, numVocab: Int, numOffsets: Int, numLabels: Int, numPartsOfSpeech: Int, numFeatureStructures: Int): (KerasNet[Float], Array[Array[Int]], Array[Int], String) = {
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
      case "x" =>
        // A model for (token ++ uPoS ++ features ++ graphX ++ Node2Vec) tensor
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
        // merge all embeddings
        val embedding = Merge.merge(inputs = List(embeddingT, embeddingP, embeddingF, reshapeX1, reshapeX2), mode = "concat")

        val offsetRNN1 = Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true).setName("LSTM-o1")).inputs(embedding)
        val offsetRNN2 = Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true).setName("LSTM-o2")).inputs(offsetRNN1)
        val labelRNN1 = Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true).setName("LSTM-d1")).inputs(embedding)
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

        val bigdl = Model(Array(inputT, inputP, inputF, inputX1, inputX2), duplicate)
        val (featureSize, labelSize) = (Array(Array(config.maxSeqLen), Array(config.maxSeqLen), Array(config.maxSeqLen), Array(3*config.maxSeqLen), Array(32*config.maxSeqLen)), Array(2*config.maxSeqLen))
        (bigdl, featureSize, labelSize, "t+p+f+x")
      case "b" => 
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
        // pass the bert output to two branches for offset and label separately
        val offsetOutput = Dense(numOffsets, activation = "softmax").setName("denseO").inputs(bertOutput)
        val labelOutput =  Dense(numLabels, activation = "softmax").setName("denseD").inputs(bertOutput)
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

        val bigdl = Model(input, duplicate)
        val (featureSize, labelSize) = (Array(Array(4*config.maxSeqLen)), Array(2*config.maxSeqLen))
        (bigdl, featureSize, labelSize, "b")
      case "bx" => 
        // A model for (token ++ uPoS ++ features ++ graphX ++ Node2Vec) tensor using BERT 
        // BERT model using one input of 4*maxSeqLen elements
        val inputT = Input(inputShape = Shape(4*config.maxSeqLen), name = "inputT")
        val reshape = Reshape(targetShape = Array(4, config.maxSeqLen)).inputs(inputT)
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
        // inputs for parts-of-speech and features
        val inputP = Input(inputShape = Shape(config.maxSeqLen), name = "inputP") 
        val inputF = Input(inputShape = Shape(config.maxSeqLen), name = "inputF") 
        val embeddingP = Embedding(numPartsOfSpeech + 1, config.partsOfSpeechEmbeddingSize).setName("posEmbedding").inputs(inputP)
        val embeddingF = Embedding(numFeatureStructures + 1, config.featureStructureEmbeddingSize).setName("fsEmbedding").inputs(inputF)
        // graphX input
        val inputX1 = Input(inputShape = Shape(3*config.maxSeqLen), name = "inputX1") // 3 for graphX
        val reshapeX1 = Reshape(targetShape = Array(config.maxSeqLen, 3)).setName("reshapeX1").inputs(inputX1)
        // Node2Vec input
        val inputX2 = Input(inputShape = Shape(32*config.maxSeqLen), name = "inputX2") // 32 for node2vec
        val reshapeX2 = Reshape(targetShape = Array(config.maxSeqLen, 32)).setName("reshapeX2").inputs(inputX2)
        // merge these two exra embeddings        
        val extra = Merge.merge(inputs = List(embeddingP, embeddingF, reshapeX1, reshapeX2), mode = "concat")
        // pass these embeddings through a RNN 
        val pfgRNN = Bidirectional(LSTM(outputDim = 32, returnSequences = true).setName("LSTM-pfg")).inputs(extra) // 32 is a magic number
        // merge bert ++ pfgRNN
        val merge = Merge.merge(inputs = List(bertOutput, pfgRNN), mode = "concat")
        // create 2 output branches for offsets and labels
        val offsetOutput = Dense(numOffsets, activation = "softmax").setName("denseO").inputs(merge)
        val labelOutput =  Dense(numLabels, activation = "softmax").setName("denseD").inputs(merge)
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
        val merge2 = if (numOffsets < numLabels) {
          Merge.merge(inputs = List(transposeBack, labelOutput), mode = "concat")
        } else {
          Merge.merge(inputs = List(offsetOutput, transposeBack), mode = "concat")
        }
        // output size is maxSeqLen x (2*max(numOffsets, numLabels))
        // duplicate to have the same seqLen as labels (for TimeDistributedCriterion to work)
        val duplicate = Merge.merge(inputs = List(merge2, merge2), mode = "concat", concatAxis = 1)

        val bigdl = Model(Array(inputT, inputP, inputF, inputX1, inputX2), duplicate)
        val (featureSize, labelSize) = (Array(Array(4*config.maxSeqLen), Array(config.maxSeqLen), Array(config.maxSeqLen), Array(3*config.maxSeqLen), Array(32*config.maxSeqLen)), Array(2*config.maxSeqLen))
        (bigdl, featureSize, labelSize, "b+p+f+x")
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
        val featureStructures = preprocessor.stages(4).asInstanceOf[CountVectorizerModel].vocabulary
        val featureStructureMap = featureStructures.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
        val numFeatureStructures = featureStructures.length

        println("   offsets = " + offsets.mkString(", "))
        println("#(offsets) = " + numOffsets)
        println("    labels = " + dependencies.mkString(", "))
        println(" #(labels) = " + numDependencies)
        println("  #(vocab) = " + numVocab)
        println("    #(PoS) = " + numPartsOfSpeech)
        println("    #(fs) = " + numFeatureStructures)
        // 
        // sequencer tokens, pos, offsets and labels; then concatenate offsets and labels
        val tokenSequencer = new Sequencer(tokensMap, config.maxSeqLen, 0f).setInputCol("tokens").setOutputCol("t")
        val posSequencer = new Sequencer(partsOfSpeechMap, config.maxSeqLen, 0f).setInputCol("uPoS").setOutputCol("p")
        val offsetSequencer = new Sequencer(offsetMap, config.maxSeqLen, -1f).setInputCol("offsets").setOutputCol("o")
        val labelSequencer = new Sequencer(dependencyMap, config.maxSeqLen, -1f).setInputCol("labels").setOutputCol("d")
        val outputAssembler = new VectorAssembler().setInputCols(Array("o", "d")).setOutputCol("o+d")
        val labelShifter = new Shifter(numOffsets, -1f).setInputCol("d").setOutputCol("d1")
        val outputAssembler1 = new VectorAssembler().setInputCols(Array("o", "d1")).setOutputCol("o+d1")
        val featureSequencer = new Sequencer(featureStructureMap, config.maxSeqLen, 0f).setInputCol("featureStructure").setOutputCol("f")
        val featureAssembler = new VectorAssembler().setInputCols(Array("t", "p", "f")).setOutputCol("t+p+f")
        val pipeline = new Pipeline().setStages(Array(tokenSequencer, posSequencer, offsetSequencer, labelSequencer, labelShifter, outputAssembler, outputAssembler1, featureSequencer, featureAssembler))
        val pipelineModel = pipeline.fit(df)

        val (ff, ffV, ffW) = (pipelineModel.transform(ef), pipelineModel.transform(efV), pipelineModel.transform(efW))
        ff.select("o+d", "o+d1").show(5, false)

        val (uf, vf, wf) = config.modelType match {
          case "t" => (ff, ffV, ffW)
          case "x" => 
            val Array(hf, hfV, hfW) = DEPx.addNodeEmbeddingFeatures(spark, config, Array(ff, ffV, ffW))
            (hf, hfV, hfW)
          case "b" =>
            val hf = ff.withColumn("b", DEPx.createInputBERT(col("t")))
            val hfV = ffV.withColumn("b", DEPx.createInputBERT(col("t")))
            val hfW = ffW.withColumn("b", DEPx.createInputBERT(col("t")))
            (hf, hfV, hfW)
          case "bx" =>
            val hf = ff.withColumn("b", DEPx.createInputBERT(col("t")))
            val hfV = ffV.withColumn("b", DEPx.createInputBERT(col("t")))
            val hfW = ffW.withColumn("b", DEPx.createInputBERT(col("t")))
            val Array(jf, jfV, jfW) = DEPx.addNodeEmbeddingFeatures(spark, config, Array(hf, hfV, hfW))
            (jf, jfV, jfW)
        }
        val (bigdl, featureSize, labelSize, featureColName) = createBigDL(config, numVocab, numOffsets, numDependencies, numPartsOfSpeech, numFeatureStructures)
        val numCores = Runtime.getRuntime().availableProcessors()
        val batchSize = if (config.batchSize % numCores != 0) numCores * 4; else config.batchSize
        val modelPath = s"${config.modelPath}/${config.language}-${config.modelType}.bigdl"
        val loss = ClassNLLCriterion(sizeAverage = false, logProbAsInput = false, paddingValue = -1)
        val criterion = TimeDistributedMaskCriterion(loss, paddingValue = -1)
        // eval() needs an array of feature column names for a proper reshaping of input tensors
        val featureColNames = config.modelType match {
          case "t" => Array("t")
          case "b" => Array("b")
          case "x" => Array("t", "p", "f", "x1", "x2")
          case "bx" => Array("b", "p", "f", "x1", "x2")
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
            Files.write(Paths.get(s"${config.scorePath}-lop.tsv"), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
          case "eval" =>
            println(s"Loading model in the path: $modelPath...")
            val bigdl = Models.loadModel[Float](modelPath)
            bigdl.summary()
            val scores = eval(bigdl, config, uf, vf, Array(featureColName))

            val heads = if (config.modelType != "b") 0 else config.heads
            val result = f"\n${config.language};${config.modelType};${config.tokenEmbeddingSize};${config.tokenHiddenSize};${config.layers};$heads;${scores(0)}%.4g;${scores(1)}%.4g"
            println(result)
            Files.write(Paths.get(s"${config.scorePath}-lop.tsv"), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
          case "validate" => 
            // perform a series of experiments to find the best hyper-params on the development set for a language
            // The arguments are: -l <lang> -t <modelType> -m validate
            val ws = Array(128, 200)
            val hs = Array(128, 200, 300, 400)
            for (_ <- 1 to 3) {
              for (w <- ws; h <- hs) {
                val cfg = config.copy(tokenEmbeddingSize = w, tokenHiddenSize = h)
                println(cfg)
                val (bigdl, featureSize, labelSize, featureColName) = createBigDL(cfg, numVocab, numOffsets, numDependencies, numPartsOfSpeech, numFeatureStructures)
                val estimator = NNEstimator(bigdl, criterion, featureSize, labelSize)
                estimator.setLabelCol("o+d1").setFeaturesCol(featureColName)
                  .setBatchSize(batchSize)
                  .setOptimMethod(new Adam(config.learningRate))
                  .setEndWhen(Trigger.or(Trigger.maxEpoch(config.epochs), Trigger.maxIteration(maxIterations)))
                // train
                estimator.fit(uf)
                val scores = eval(bigdl, cfg, uf, vf, featureColNames)
                val result = f"\n${cfg.language};${cfg.modelType};$w;$h;2;0;${scores(0)}%.4g;${scores(1)}%.4g"
                Files.write(Paths.get(s"${config.scorePath}-lop.tsv"), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
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
                val (bigdl, featureSize, labelSize, featureColName) = createBigDL(cfg, numVocab, numOffsets, numDependencies, numPartsOfSpeech, numFeatureStructures)
                val estimator = NNEstimator(bigdl, criterion, featureSize, labelSize)
                estimator.setLabelCol("o+d1").setFeaturesCol(featureColName)
                  .setBatchSize(batchSize)
                  .setOptimMethod(new Adam(config.learningRate))
                  .setEndWhen(Trigger.or(Trigger.maxEpoch(config.epochs), Trigger.maxIteration(maxIterations)))
                // train
                estimator.fit(uf)
                val scores = eval(bigdl, cfg, uf, vf, featureColNames)
                val result = f"\n${cfg.language};${cfg.modelType};$w;$h;$j;$n;${scores(0)}%.4g;${scores(1)}%.4g"
                Files.write(Paths.get(s"${config.scorePath}-lop.tsv"), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
              }
            }
          }
      case None => {}
    }
  }
}
