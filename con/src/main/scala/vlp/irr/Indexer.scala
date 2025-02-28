package vlp.irr

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg

/**
  * (C) phuonglh@gmail.com
  * 
  * Indexing a large corpus using Apache Spark.
  * 
  */
object Indexer {
  def main(args: Array[String]): Unit = {
    val path = "/home/phuonglh/Downloads/c4-train.00000-of-01024-30K.json.gz"
    val spark = SparkSession.builder().master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val df = spark.read.json(path)
    val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern("""[\s#/.,;?!:“”‘’"'…▲\-)(\]\[*+_]+""")
    val ef = tokenizer.transform(df)
    val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("vector").setMinDF(5)
    val model = vectorizer.fit(ef)
    val vocabulary = model.vocabulary
    println(s"Size of vocab = ${vocabulary.size}.")

    val gf = model.transform(ef)

    val f = udf((v: linalg.SparseVector) => v.indices.toSet) 
    val hf = gf.withColumn("indices", f(col("vector")))
    hf.printSchema()
    hf.select("url", "indices").show(5)

    val g = udf((u: String, xs: Seq[Int]) => xs.map((u, _)))
    val uf = hf.withColumn("pairs", g(col("url"), col("indices")))

    val vf = uf.select(explode(col("pairs")).alias("p"))
    vf.show(false)
    vf.printSchema()

    val yf = vf.groupBy("p._2").agg(collect_set("p._1"))
    yf.show(2, false)
    println(yf.count)

    spark.stop()
  }
}
