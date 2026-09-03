package vlp.sub

import org.apache.spark.sql.SparkSession

object Wikipedia {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[4]")
      .config("spark.driver.memory", "24g")
      .appName(getClass().getName())
      .getOrCreate()

    val df = spark.read.parquet("/home/phuonglh/code/con/src/main/python/20231101_eng.parquet")

    import spark.implicits._
    val ef = df.select("text").flatMap { row => 
      val text = row.getString(0)
      text.split("""\n+""").map(_.trim())
        .filter(_.length() >= 20)   // two short lines
        .filter(!_.startsWith("|")) // table rows
        .filter(!_.startsWith("{|")) // table rows
        .filter(!_.contains("wikitable"))
    }
    
    ef.repartition(10).write.text("/home/phuonglh/code/con/src/main/python/ptok/20231101_eng")
    
    spark.stop()
  }
}