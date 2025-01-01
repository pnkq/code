package vlp.woz.nlu

import org.apache.spark.sql.{SparkConf, SparkContext, SparkSession}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature._

/**
  * Reads dialog act data sets which are saved by [[vlp.woz.DialogReader]] and prepare 
  * data sets suitable for training token classification (sequence labeling) models.
  * 
  */
object NLU {

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
    val tokenizer = new RegexTokenizer().setInputCol("utterance").setOutputCol("tokens").setPattern("""[?.,!]+""")
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName(getClass().getName()).setMaster("local[*]")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    sc.setLogLevel("ERROR")

    spark.stop()
  }
}
