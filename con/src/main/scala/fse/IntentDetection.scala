package fse

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification._
import org.apache.spark.ml._
import org.apache.spark.ml.evaluation._


object IntentDetection {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val train = spark.read.option("header", true).csv("dat/hwu/train.csv")
    val test = spark.read.option("header", true).csv("dat/hwu/test.csv")
    train.show(false)

    val indexer = new StringIndexer().setInputCol("category").setOutputCol("label")
    val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("tokens")
      .setPattern("""[\s,.?]+""")
    val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("v")
    val idf = new IDF().setInputCol("v").setOutputCol("features")
    val classifier = new LogisticRegression()

    val pipeline = new Pipeline().setStages(Array(indexer, tokenizer, vectorizer, idf, classifier))

    val pipelineModel = pipeline.fit(train)
    val df = pipelineModel.transform(train)
    val ef = pipelineModel.transform(test)

    df.select("features").show(5)
    df.select("label", "prediction").show(20)

    val evaluator = new MulticlassClassificationEvaluator()
    val trainingScore = evaluator.evaluate(df)
    val testScore = evaluator.evaluate(ef)

    println(s"trainingScore = $trainingScore, testScore = $testScore.")
    spark.stop()
  }
}