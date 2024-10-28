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
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.{Trigger, Loss}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}

import vlp.ner.{ArgMaxLayer, Sequencer, TimeDistributedTop1Accuracy}


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
        // 1. Sequential model with random token embeddings
        val bigdl = Sequential()
        bigdl.add(Embedding(numVocab + 1, config.tokenEmbeddingSize, inputLength = config.maxSeqLen).setName("tokEmbedding"))
        for (_ <- 1 to config.layers)
          bigdl.add(Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true)))
        bigdl.add(Dropout(config.dropoutRate))
        bigdl.add(Dense(numOffsets + numLabels, activation = "softmax"))
        val (featureSize, labelSize) = (Array(Array(config.maxSeqLen)), Array(2*config.maxSeqLen))
        (bigdl, featureSize, labelSize, "t")
      case "b" => 
        val bigdl = Sequential()
        val (featureSize, labelSize) = (Array(Array(config.maxSeqLen)), Array(2*config.maxSeqLen))
        (bigdl, featureSize, labelSize, "t")
    }
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
        val pipeline = new Pipeline().setStages(Array(tokenSequencer, posSequencer, offsetSequencer, labelSequencer, outputAssembler))
        val pipelineModel = pipeline.fit(df)

        val (ff, ffV, ffW) = (pipelineModel.transform(ef), pipelineModel.transform(efV), pipelineModel.transform(efW))
        ff.show(5)

      case None => {}
    }
  }
}
