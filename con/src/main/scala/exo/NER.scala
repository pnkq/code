package exo

import scala.collection.mutable.{ListBuffer, Map}
import scala.io.Source
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

/**
  * (C) phuonglh@gmail.com
  * 
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

  val labelMap = Seq("O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC").zipWithIndex.map(p => (p._1, p._2 + 1)).toMap

  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName(getClass.getName).setMaster("local[*]")
      .set("spark.executor.memory", "4g").set("spark.driver.memory", "8g")
    val sc = new SparkContext(conf)
    Engine.init

    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val splits = Seq("testa", "testa", "testb").map(s => readCoNLL(s"dat/ner/eng.$s"))
    val Seq(train, dev, test) = splits.map { split =>
      val rows = split.map { sentence =>
        Row(sentence.tokens.map(_.word), sentence.tokens.map(_.partOfSpeech), sentence.tokens.map(_.chunk), sentence.tokens.map(_.entity))
      }
      spark.createDataFrame(rows.filter(_.words.size >= 3))
    }
    dev.show(5)
    dev.printSchema()
    println(s"#(train) = ${train.count()}, #(dev) = ${dev.count()}, #(test) = ${test.count()}.")

    spark.stop()
  }
}
