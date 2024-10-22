package vlp.dep

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


/**
  * Analyse parsing results; pull out max scores.
  */

object Result {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val path = "dat/depx-scores-uas-sun-3.tsv"
    val modelType = "b"
    val df = spark.read.options(Map("delimiter" -> ";", "inferSchema" -> "true")).csv(path).toDF("lang", "model", "w", "h", "j", "b", "train", "valid")
    val languages = Array("eng", "fra", "ind", "vie")
    val maxScores = languages.map { lang =>
      val ef = df.filter(col("lang") === lang && col("model") === modelType)
      val ff = ef.select("w", "h", "j", "b", "train", "valid").groupBy("w", "h", "j", "b").mean("train", "valid")
      val maxRow = ff.sort(desc("avg(valid)")).head()
      (lang, maxRow)
    }
    maxScores.foreach(println(_))

    spark.stop()
  }
}
