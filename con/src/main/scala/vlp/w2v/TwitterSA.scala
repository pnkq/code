package vlp.w2v

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{RegexTokenizer, StringIndexer, CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.SparkSession
import scala.io.Source

/**
 * phuonglh@gmail.com
 *
 */

object TwitterSA {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[4]").config("spark.driver.memory", "16g").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    // read the sentiment dataset
    val dataPath = "dat/sentiments.csv"
    // val wordVectorPath = "dat/glove.6B.50d.txt"
    val wordVectorPath = "/opt/data/emb/en/glove.6B.50d.txt"
    val df = spark.read.option("header", true).csv(dataPath).na.drop()
    val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2), seed = 2207)
    // load a map of GloVe embeddings
    val dictionary = Source.fromFile(wordVectorPath).getLines().map { line =>
      val parts = line.split("\\s+")
      val word = parts.head
      val values = parts.tail.map(_.toDouble).toSeq
      (word, values)
    }.toMap
    println("#(GloVe) = " + dictionary.size)

    // create a pre-processing pipeline
    val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern("""[\s.,;?!:"'▲\-)(\]\[]+""")
    val counter = new CountVectorizer().setInputCol("tokens").setOutputCol("vector").setMinDF(2)
    val preprocessor = new Pipeline().setStages(Array(tokenizer, counter))
    val preprocessorModel = preprocessor.fit(trainingData)
    val vocabulary = preprocessorModel.stages(1).asInstanceOf[CountVectorizerModel].vocabulary.toSet
    println(s"#vocabSize = ${vocabulary.size}")
    val reducedDictionary = dictionary.filter(p => vocabulary.contains(p._1))

    val indexer = new StringIndexer().setInputCol("sentiment").setOutputCol("label")
    val embedding = new PretrainedEmbedding(reducedDictionary, 50).setInputCol("tokens").setOutputCol("features")
    val classifier = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(indexer, tokenizer, embedding, classifier))
    val model = pipeline.fit(trainingData)
    // model.save("bin/tsa")
    val predictionTest = model.transform(testData)
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")

    val testScore = evaluator.evaluate(predictionTest)
    println("testScore = " + testScore)

    spark.stop()
  }
}
