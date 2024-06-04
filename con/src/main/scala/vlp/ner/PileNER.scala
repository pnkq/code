package vlp.ner

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object PileNER {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[*]").getOrCreate()
    val df = spark.read.option("multiline", true).json("dat/pile-ner-type.json.gz")
    val ef = df.select(explode(col("conversations")))
    val ff = ef.select("col.from", "col.value")
    // the "from" column contains only two values ["human", "gpt"].
    // select human rows for the template "What describes <entityType> in the text?", ignore "gpt" rows
    val human = ff.filter(col("from") === "human") // 404,070 rows
    // filter rows starting with "What"
    val what = human.filter(col("value").startsWith("What"))
    
    spark.stop()
  }
}
