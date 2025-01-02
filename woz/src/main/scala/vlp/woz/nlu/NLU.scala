package vlp.woz.nlu

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature._

/**
  * Reads dialog act data sets which are saved by [[vlp.woz.DialogReader]] and prepare 
  * data sets suitable for training token classification (sequence labeling) models.
  * 
  */
object NLU {

  def tokenize(utterance: String, intervals: List[(Int, Int)]): Seq[(Int, Int, Array[String])] = {
    val (a, b) = (intervals.head._1, intervals.last._2)
    // build intervals that need to be tokenized
    val js = new collection.mutable.ArrayBuffer[(Int, Int)](intervals.size + 1)
    js.append((0, a))
    for (j <- 0 until intervals.size - 1) {
      js.append((intervals(j)._2, intervals(j+1)._1))
    }
    js.append((b, utterance.size))
    // build results
    val ss = new collection.mutable.ArrayBuffer[(Int, Int, Array[String])](intervals.size*2)
    intervals.foreach(p => ss.append((p._1, p._2, Array(utterance.subSequence(p._1, p._2).toString()))))
    js.foreach { p => 
      val text = utterance.subSequence(p._1, p._2).toString().trim()
      val tokens = text.split("""[?.,!\s]+""").filter(_.nonEmpty)
      ss.append((p._1, p._2, tokens))
    }
    ss.toSeq.sortBy(_._1)
  }
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
  def readActDataset(spark: SparkSession, path: String): DataFrame = {
    val af = spark.read.json(path)
    af
  }

  def main(args: Array[String]): Unit = {
    // val conf = new SparkConf().setAppName(getClass().getName()).setMaster("local[*]")
    // val sc = new SparkContext(conf)
    // val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    // sc.setLogLevel("ERROR")

    val utterance = "Sheep's Green and Lammas Land Park Fen Causeway is a park in the South part of town at Fen Causeway, Newnham Road. The postcoe is CB22AD."
    val intervals = List((35, 47), (65, 70), (101, 113), (130, 136))
    val js = tokenize(utterance, intervals)
    js.foreach(a => println(a._1, a._2, a._3.mkString("|")))
    // {"dialogId":"PMUL4688.json","turnId":"5","utterance":,"acts":[{"actName":"Attraction-Inform","slot":"address","value":"Fen Causeway","start":35,"end":47},{"actName":"Attraction-Inform","slot":"area","value":"South","start":65,"end":70},{"actName":"Attraction-Inform","slot":"address","value":"Newnham Road","start":101,"end":113},{"actName":"Attraction-Inform","slot":"postcode","value":"CB22AD","start":130,"end":136}]}
    // spark.stop()
  }
}
