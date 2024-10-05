package vlp.w2v

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.Word2VecModel

object Word2Vec {
  def main(args: Array[String]): Unit = {
    val path = "/home/phuonglh/Downloads/c4-train.00000-of-01024-30K.json"
    val spark = SparkSession.builder().master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val df = spark.read.json(path)
    val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern("""[\s.,;?!:"'▲\-)(\]\[]+""")
    val ef = tokenizer.transform(df)
    // val w2v = new feature.Word2Vec().setInputCol("tokens").setOutputCol("result").setVectorSize(10).setMinCount(3)
    // val model = w2v.fit(ef)
    // val wf = model.getVectors
    // model.save("bin/w2v/")

    // wf.show(20, false)

    val model = Word2VecModel.load("bin/w2v/")
    val wf = model.getVectors
    wf.show(20)
    // println(s"Number of words = ${wf.count}")

    val ff = model.transform(ef)
    ff.show()

    ff.printSchema()

    spark.stop()
  }
}
