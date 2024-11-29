package vlp.ecc

import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization

object ECC {
  /**
    * Remove a prefix string (before the first ':'), for example "claim:" or "premises:".
    *
    * @param text a given text
    * @return a text with prefix removed.
    */
  def removePrefix(text: String): String = {
    val j = text.indexOf(':')
    if (j < 0) text else text.substring(j+1).trim
  }

  val f = udf((text: String) => removePrefix(text))

  implicit val formats: Formats = Serialization.formats(NoTypeHints)

  def extractPremises(premises: String): List[String] = {
    val text = premises.replaceAll(", '", ", \"")
      .replaceAll("', ", "\", ")
      .replaceAll("""\['""", """\["""")
      .replaceAll("""'\]""", """"\]""")

    parse(text).extract[List[String]]
  }

  val g = udf((text: String) => extractPremises(text))

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName(ECC.getClass().getName()).master("local[4]").getOrCreate()

    val path = "dat/ecc/ECC-val.tsv"
    val df = spark.read.options(Map("delimiter" -> "\t", "header" -> "true")).csv(path)
    val ef = df.withColumn("claim", f(col("claim_text"))).withColumn("premises", f(col("premise_texts")))
    ef.show(5)

    val gf = ef.withColumn("xs", g(col("premises")))
    gf.show(5)

    val hf = gf.select("claim", "year", "quarter", "label", "xs").withColumn("premise", explode(col("xs")))
    hf.show(20)
    hf.printSchema()
    println(s"Number of samples = ${hf.count}.")


    spark.stop()
  }
}
