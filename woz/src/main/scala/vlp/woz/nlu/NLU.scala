package vlp.woz.nlu

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature._

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._

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
  acts: Array[Span]
)

/**
  * Reads dialog act data sets which are saved by [[vlp.woz.DialogReader]] and prepare 
  * data sets suitable for training token classification (sequence labeling) models.
  * 
  */
object NLU {

  val patterns = """[?.,!\s]+"""

  def tokenize(utterance: String, spans: Array[Span]): Seq[(Int, Array[(String, String)])] = {
    val intervals: Array[(Int, Int)] = spans.map { span => (span.start.get.toInt, span.end.get.toInt) }
    val (a, b) = (intervals.head._1, intervals.last._2)
    // build intervals that need to be tokenized
    val js = new collection.mutable.ArrayBuffer[(Int, Int)](intervals.size + 1)
    if (a > 0) js.append((0, a))
    for (j <- 0 until intervals.size - 1) {
      js.append((intervals(j)._2, intervals(j+1)._1))
    }
    if (b < utterance.size) js.append((b, utterance.size))
    // build results
    val ss = new collection.mutable.ArrayBuffer[(Int, Array[(String, String)])](intervals.size*2)
    for (j <- 0 until intervals.size) {
      val p = intervals(j)
      val slot = spans(j).slot
      val value = utterance.subSequence(p._1, p._2).toString().trim()
      val tokens = value.split(patterns)
      val labels = s"B-${slot.toUpperCase()}" +: Array.fill[String](tokens.size-1)(s"I-${slot.toUpperCase()}")
      ss.append((p._1, tokens.zip(labels)))
    }
    js.foreach { p => 
      val text = utterance.subSequence(p._1, p._2).toString().trim()
      if (text.size > 0) { 
        val tokens = text.split(patterns).filter(_.nonEmpty)
        ss.append((p._1, tokens.zip(Array.fill[String](tokens.size)("O"))))
      }
    }
    ss.toSeq.sortBy(_._1)
  }

  val f = udf((utterance: String, spans: Array[Span]) => {
    tokenize(utterance, spans)
  })

  /**
    * Reads a data set and creates a df of columns (utterance, tokenSequence, slotSequence, actNameSequence), where
    * <ol>
    * <li>utterance: String, is a original text</li>
    * <li>tokenSequence: Seq[String], is a sequence of tokens from utterance</li>
    * <li>slotSequence: Seq[String], is a sequence of slot names (entity types, in the form of B/I/O)</li>
    * <li>actNameSequence: Seq[String], is a sequence of act names, which is typically 1 or 2 act names.
    * </ol>
    *
    * @param spark
    * @param path
    */
  def transformActs(spark: SparkSession, path: String): DataFrame = {
    import spark.implicits._
    val af = spark.read.json(path).as[Element]
    println("Number of rows = " + af.count())
    val bf = af.filter(size(col("acts")) > 0)
    println("Number of non-empty rows = " + bf.count())
    val cf = bf.withColumn("seq", f(col("utterance"), col("acts")))
    cf
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName(getClass().getName()).setMaster("local[*]")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    sc.setLogLevel("ERROR")
    
    val df = transformActs(spark, "dat/woz/act/dev")
    df.select("seq").show(false)
    df.printSchema()

    spark.stop()
  }
}
