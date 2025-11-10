package exo

import scala.collection.mutable.{ListBuffer, Map}
import scala.io.Source
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import com.intel.analytics.bigdl.dllib.nn.TimeDistributedMaskCriterion
import com.intel.analytics.bigdl.dllib.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.visualization.TrainSummary
import com.intel.analytics.bigdl.dllib.visualization.ValidationSummary
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.optim.Trigger
import vlp.dep.TimeDistributedTop1Accuracy
import com.intel.analytics.bigdl.dllib.nnframes.NNModel
import com.intel.analytics.bigdl.dllib.keras.models.KerasNet
import vlp.ner.ArgMaxLayer


/**
  * (C) phuonglh@gmail.com
  * 
  * Starter code for NER, using words as features for a bidirectional LSTM model with BigDL.
  * Course: NLP & DL at VNU-HUS.
*/

object Label extends Enumeration {
  val PartOfSpeech, Chunk, NamedEntity = Value 
}

case class Token(word: String, annotation: Map[Label.Value, String]) {
  override def toString(): String = {
    val s = new StringBuilder()
    s.append("Token(")
    s.append(word)
    s.append(",[")
    if (!annotation.keys.isEmpty) {
      val a = new StringBuilder()
      annotation.keys.foreach { 
        k => {
          a.append(k.toString)
          a.append("=")
          a.append(annotation(k)) 
          a.append(' ')
        }
      }
      s.append(a.toString.trim)
    }
    s.append("])")
    s.toString()
  }
  
  def chunk: String = annotation.getOrElse(Label.Chunk, None.toString)
  def partOfSpeech: String = annotation.getOrElse(Label.PartOfSpeech, None.toString)
  def entity: String = annotation.getOrElse(Label.NamedEntity, None.toString)
  def setEntity(entity: String): Unit = annotation += (Label.NamedEntity -> entity)
}

case class Sentence(tokens: Seq[Token])
case class Row(words: Seq[String], tags: Seq[String], chunks: Seq[String], entities: Seq[String])

object NER {
  val map = Seq("O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC").zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
  
  val index = udf((labels: Array[String], maxSeqLen: Int) => {
    val ys = labels.map(token => map(token))
    if (ys.length < maxSeqLen) ys ++ Array.fill[Int](maxSeqLen - ys.length)(-1) else ys.take(maxSeqLen)
  })

  /**
    * Reads a NER corpus in CoNLL-2003 format.
    * @param path
    * @return a list of sentences.
    */
  def readCoNLL(path: String): Seq[Sentence] = {
    val lines = (Source.fromFile(path, "UTF-8").getLines() ++ List("")).toArray
    val sentences = new ListBuffer[Sentence]()
    val indices = lines.zipWithIndex.filter(p => p._1.trim.isEmpty).map(p => p._2)
    var u = 0
    var v = 0
    for (i <- (0 until indices.length)) {
      v = indices(i)
      if (v > u) { // don't treat two consecutive empty lines
        val s = lines.slice(u, v)
        val tokens = s.map(line => {
          val parts = line.trim.split("\\s+")
          Token(parts(0), Map(Label.PartOfSpeech -> parts(1), Label.Chunk -> parts(2), Label.NamedEntity -> parts(3)))
        })
        sentences.append(Sentence(tokens.toList.to[ListBuffer]))
      }
      u = v + 1
    }
    sentences.toList
  }

  def fit(model: Sequential[Float], maxSeqLen: Int, uf: DataFrame, vf: DataFrame): NNModel[Float] = {
    val criterion = TimeDistributedMaskCriterion(
      ClassNLLCriterion(sizeAverage = false, logProbAsInput = false, paddingValue = -1), 
      paddingValue = -1
    )
    val estimator = NNEstimator(model, criterion, Array(maxSeqLen), Array(maxSeqLen))
    val trainingSummary = TrainSummary(appName = "lstm", logDir = "sum/ner/")
    val validationSummary = ValidationSummary(appName = "lstm", logDir = "sum/ner/")
    val batchSize = Runtime.getRuntime.availableProcessors() * 8
    estimator.setLabelCol("label").setFeaturesCol("features")
      .setBatchSize(batchSize)
      .setOptimMethod(new Adam(2E-4))
      .setMaxEpoch(25)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, vf, Array(new TimeDistributedTop1Accuracy(-1)), batchSize)
    val nn = estimator.fit(uf)
    // model.saveModel("bin/ner.bigdl", overWrite = true)
    nn.save("bin/ner")
    return nn
  }

  def transform(nn: NNModel[_], df: DataFrame) = {
    val sequential = nn.getModel().asInstanceOf[Sequential[Float]]
    // bigdl produces 3-d output results (including batch dimension), we need to convert it to 2-d results.
    sequential.add(ArgMaxLayer())
    sequential.summary()
    // wrap to a Spark model and run prediction
    val model = NNModel(sequential)
    val output = model.transform(df)
    output.select("words", "entities", "prediction").show(10, false)
  }


  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName(getClass.getName).setMaster("local[*]")
      .set("spark.executor.memory", "4g").set("spark.driver.memory", "8g")
    val sc = new SparkContext(conf)
    Engine.init

    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val splits = Seq("train", "testa", "testb").map(s => readCoNLL(s"dat/ner/eng.$s"))
    val Seq(train, valid, test) = splits.map { split =>
      val rows = split.map { sentence =>
        Row(sentence.tokens.map(_.word), sentence.tokens.map(_.partOfSpeech), sentence.tokens.map(_.chunk), sentence.tokens.map(_.entity))
      }
      spark.createDataFrame(rows.filter(_.words.size >= 3))
    }
    valid.show(5)
    valid.printSchema()
    println(s"#(train) = ${train.count()}, #(valid) = ${valid.count()}, #(test) = ${test.count()}.")

    val vocabSize = 8192*2
    val maxSeqLen = 30

    val uf = train.withColumn("features", Tagger.hash(col("words"), lit(vocabSize), lit(maxSeqLen)))
      .withColumn("label", index(col("entities"), lit(30)))
    val vf = valid.withColumn("features", Tagger.hash(col("words"), lit(vocabSize), lit(maxSeqLen)))
      .withColumn("label", index(col("entities"), lit(30)))
    vf.select("features", "label").show(5, false)

    val model = Tagger.createModel(maxSeqLen, vocabSize, 50, map.size)
    model.summary()

    val nn = fit(model, maxSeqLen, uf, vf)
    // val nn: NNModel[_] = NNModel.load("bin/ner")
    // transform(nn, vf)

    spark.stop()
  }
}
