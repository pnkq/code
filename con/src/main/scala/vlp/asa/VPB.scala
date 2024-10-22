package vlp.asa

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{RegexTokenizer, CountVectorizer}
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

/**
  * phuonglh@gmail.com
  * 
  */

case class Turn(channel: String, start: Double, end: Double, trans: String)
case class Call(call_name: String, file_name: String, conversation: Array[Turn])

object VPB {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName(getClass().getName()).setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    import spark.implicits._

    val path = "/home/phuonglh/code/con/dat/MergedChannel_20240924_Collections_Result.json"

    val df = spark.read.options(Map("multiline" -> "true", "inferSchema" -> "true")).json(path).as[Call]
    df.show(5)
    df.printSchema()
    println("#(conversations) = " + df.count)

    val ef = df.select("conversation.trans")
    ef.show()
    println("#(turns) = " + ef.count())

    // merge consecutive turns of the same channel into one turn, using '|'' to concatenate text
    val mergeTurn = udf((ts: Array[Turn]) => {
      val output = scala.collection.mutable.ArrayBuffer[Turn]()
      val n = ts.size
      var i = 0
      while (i < n-1) {
        var j = i+1
        val channel = ts(i).channel
        while ((j < n-1) && (channel == ts(j).channel)) { j = j + 1  }
        val ys = ts.slice(i, j)
        val text = ys.map(_.trans.trim()).filter(_.nonEmpty).mkString("|")
        output.append(Turn(channel, ts(i).start, ts(j-1).end, text))
        i = j
      }
      output.toArray
    })

    val ff = df.withColumn("merged", mergeTurn(col("conversation")))
    ff.show()
    println(ff.select("merged").head)

    // extract bot text from each conversation
    val extractBotText = udf((ts: Array[Turn]) => {
      ts.filter(p => p.channel == "Bot").map(_.trans.trim).filter(_.nonEmpty).mkString("|")
    })
    val bf = ff.withColumn("bot", extractBotText(col("merged")))
    bf.show(5)
    println(bf.select("bot").head)

    // extract label from file_name
    val extractLabel = udf((x: String) => {
      if (x.contains("goodcall")) 0d else 1d
    })
    val cf = bf.withColumn("label", extractLabel(col("file_name")))
    cf.select("bot", "label").show
    cf.groupBy("label").count.show()

    // create a classification pipeline
    val tokenizer = new RegexTokenizer().setInputCol("bot").setOutputCol("tokens").setPattern("""[\s|]+""")
    val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("vector").setMinDF(2)
    val idf = new IDF().setInputCol("vector").setOutputCol("features")
    val classifier = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(tokenizer, vectorizer, idf, classifier))

    val Array(training, test) = cf.randomSplit(Array(0.8, 0.2), seed=1234)
    val model = pipeline.fit(training)
    val vf = model.transform(test)
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
    val score = evaluator.evaluate(vf)
    println(s"score = $score")

    spark.stop()
  }
}
