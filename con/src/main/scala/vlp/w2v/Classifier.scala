package vlp.w2v

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, BinaryClassificationEvaluator}
import org.apache.spark.ml.classification.LogisticRegression


object Classifier {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val df = spark.read.option("header", true).option("delimiter", ";").csv("/Users/phuonglh/Downloads/nlu.csv.gz").na.drop()
    val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2), seed=1234)
    val indexer = new StringIndexer().setInputCol("intent").setOutputCol("label")
    val tokenizer = new Tokenizer().setInputCol("answer_normalised").setOutputCol("tokens")
    val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("v")
    val classifier = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(indexer, tokenizer, vectorizer))
    val model = pipeline.fit(trainingData)
    val vocab = model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
    print(vocab.mkString(", "))

    // load a map of GloVe embeddings
    val wordVectorPath = "/opt/data/emb/en/glove.6B.50d.txt"
    val dictionary = scala.io.Source.fromFile(wordVectorPath).getLines().map { line =>
      val parts = line.split("\\s+")
      val word = parts.head
      val values = parts.tail.map(_.toDouble).toSeq
      (word, values)
    }.toMap
    println("#(GloVe) = " + dictionary.size)
    val smallDict = vocab.map { token =>
      (token, dictionary.getOrElse(token, Array.fill[Double](50)(0d).toSeq))
    }.toMap
    println("#(smallDict) = " + smallDict.size)

    val glove = new PretrainedEmbedding(smallDict, 50).setInputCol("tokens").setOutputCol("features")
    val pipeline3 = new Pipeline().setStages(Array(indexer, tokenizer, glove, classifier))
    val model3 = pipeline3.fit(trainingData)
    val output = model3.transform(testData)
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("f1")
    val score = evaluator.evaluate(output)
    println("GloVe score = " + score)
    spark.stop()
  }
}
