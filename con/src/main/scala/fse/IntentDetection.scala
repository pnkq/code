package fsb

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, StringIndexer, Tokenizer}
import org.apache.spark.sql.SparkSession

object IntentDetection {
  def main(args: Array[String]): Unit = {
    val trainData = "/Users/phuonglh/corpora/dialoglue/data_utils/dialoglue/hwu/train.csv"
    val spark = SparkSession.builder().master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val df = spark.read.option("header", true).csv(trainData)
    println("#(samples) = " + df.count)
    df.show(5, false)
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")
    val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("vector")
    val indexer = new StringIndexer().setInputCol("category").setOutputCol("label")
//    val classifier = new LogisticRegression().setFeaturesCol("vector").setLabelCol("label")
    val classifier = new MultilayerPerceptronClassifier().setLayers(Array(4626, 256, 64)).setFeaturesCol("vector")
    val pipeline = new Pipeline().setStages(Array(tokenizer, vectorizer, indexer, classifier))
    val model = pipeline.fit(df)
    val ef = model.transform(df)
    ef.select("label", "prediction").show(10, false)

    val evaluator = new MulticlassClassificationEvaluator()
    val score = evaluator.evaluate(ef)
    println("training score = " + score)

    val testData = "/Users/phuonglh/corpora/dialoglue/data_utils/dialoglue/hwu/test.csv"
    val tf = spark.read.option("header", true).csv(testData)
    val uf = model.transform(tf)
    val testScore = evaluator.evaluate(uf)
    println("test score = " + testScore)

    spark.stop()
  }
}
