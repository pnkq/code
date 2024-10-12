package vlp.w2v

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.LogisticRegression


object Classifier {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val df = spark.read.option("header", true).csv("/Users/phuonglh/Downloads/sentiments.csv").na.drop()
    val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2), seed=1234)
    val indexer = new StringIndexer().setInputCol("sentiment").setOutputCol("label")
    val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern("""[\s,.;'?":]""")
    val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("features").setMinDF(2)
    val classifier = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(indexer, tokenizer, vectorizer, classifier))
    val model = pipeline.fit(trainingData)
    // model.save("bin/classifier")
    val tf = model.transform(testData)
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
    val score = evaluator.evaluate(tf)
    println("test score = " + score)
    spark.stop()
  }
}
