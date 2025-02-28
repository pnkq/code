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
        val spark = SparkSession.builder().master("local[4]").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

        //    val path = "/home/phuonglh/Downloads/c4-train.00000-of-01024-30K.json.gz"
        val path = "/Users/phuonglh/Downloads/c4-train.00000-of-01024-30K.json.gz"
        val df = spark.read.json(path)
        val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern("""[\s#/.,;?!:“”‘’"'…▲\-)(\]\[*+_]+""")
        val ef = tokenizer.transform(df)
        val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("vector").setMinDF(5)
        val model = vectorizer.fit(ef)
        val vocabulary = model.vocabulary
        println(s"Size of vocab = ${vocabulary.length}.")

        val gf = model.transform(ef)

        val f1 = udf((v: linalg.SparseVector) => v.indices.toSet)
        val hf = gf.withColumn("indices", f1(col("vector")))
        hf.printSchema()
        hf.select("url", "indices").show(5)

        val f2 = udf((u: String, xs: Seq[Int]) => xs.map((u, _)))
        val uf = hf.withColumn("pairs", f2(col("url"), col("indices")))

        val vf = uf.select(explode(col("pairs")).alias("p"))
        vf.printSchema()

        val yf = vf.groupBy("p._2").agg(collect_set("p._1").alias("u"))
        val f3 = udf((i: Int) => vocabulary(i))
        val zf = yf.withColumn("t", f3(col("_2"))).select("t", "u")
        zf.show()

        zf.repartition(5).write.json("dat/idx")

        spark.stop()
    }
}
