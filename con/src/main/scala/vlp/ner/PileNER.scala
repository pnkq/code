package vlp.ner

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object PileNER {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val df = spark.read.option("multiline", true).json("dat/pile-ner-type.json.gz")
    val ef = df.select(explode(col("conversations")))
    val ff = ef.select("col.from", "col.value")
    // the "from" column contains only two values ["human", "gpt"].
    // select human rows for the template "What describes <entityType> in the text?", ignore "gpt" rows
    val human = ff.filter(col("from") === "human") // 404,070 rows
    // filter rows starting with "What"
    val what = human.filter(col("value").startsWith("What"))

    // create a user-defined function to extract entity type from the "value" colume
    val f = udf((x: String) => {
      val j = x.indexOf("in the text")
      x.substring("What describes".length, j).trim.toLowerCase
    })
    val entity = what.withColumn("entity", f(col("value")))
    entity.show(50, false)
    val uniqueEntity = entity.select("entity").distinct()
    println(uniqueEntity.count())
    uniqueEntity.repartition(1).write.text("dat/entity")
    spark.stop()
  }
}
