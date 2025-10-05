package vlp.dep

import org.apache.spark.sql.SparkSession

/**
  * phuonglh
  * 
  */

case class Sample(
  word: Seq[String],
  pos: Seq[String],
  upos: Seq[String],
  head: Seq[String],
  label: Seq[String],
  text: String
) 

object TreebankReader {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName(TreebankReader.getClass().getName()).master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("INFO")

    val splits = Seq("dev", "test", "train")
    splits.foreach { split =>
      val path = s"dat/dep/UD_English-EWT/en_ewt-ud-${split}.conllu"
      val sentences = GraphReader.read(path).filter(_.sentence.tokens.size >= 5)

      val samples = sentences.map { graph =>
        val tokens = graph.sentence.tokens.tail // bypass the ROOT token
        val word = tokens.map(token => token.word)
        val pos = tokens.map(token => token.partOfSpeech)
        val upos = tokens.map(token => token.universalPartOfSpeech)
        val head = tokens.map(token => token.head)
        val label = tokens.map(token => token.dependencyLabel)
        Sample(word, pos, upos, head, label, word.mkString(" "))
      }

      import spark.implicits._
      val df = spark.createDataset(samples)
      df.repartition(3).write.parquet(s"dat/dep/eng-$split")
    }

    spark.stop()
  }
}
