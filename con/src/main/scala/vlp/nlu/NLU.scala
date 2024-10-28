package vlp.nlu

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.PipelineModel
import scala.io.Source
import vlp.w2v.PretrainedEmbedding
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier

object NLU {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val df = spark.read.options(Map("delimiter" -> ";", "header" -> "true"))
      .csv("/Users/phuonglh/Downloads/nlu.csv").na.drop().select("intent", "answer_normalised")
    df.show(20, false)

    val indexer = new StringIndexer().setInputCol("intent").setOutputCol("label")
    val tokenizer = new Tokenizer().setInputCol("answer_normalised").setOutputCol("tokens")
    val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("features")
    val classifier = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(indexer, tokenizer, vectorizer, classifier))

    val Array(training, test) = df.randomSplit(Array(0.8, 0.2), seed=1234)

    val model = PipelineModel.load("bin/nlu")
    val vocab = model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
    println("#(vocab) = ", vocab.size)
    vocab.take(20).foreach(println)
    println()
    vocab.takeRight(20).foreach(println)

    val glove = "/opt/data/emb/en/glove.6B.50d.txt"
    val lines = Source.fromFile(glove).getLines().toList
    val gloveDict = lines.map { line => 
      val parts = line.split(" ")
      val word = parts.head
      val vector = parts.tail.map(_.toDouble).toSeq
      (word, vector)
    }.toMap

    val myDict = vocab.map { word => 
       val vector = gloveDict.getOrElse(word, Seq.fill[Double](50)(0d))
       (word, vector)
    }.toMap

    val gloveVectorizer = new PretrainedEmbedding(myDict, 50).setInputCol("tokens").setOutputCol("features")
    val classifier2 = new MultilayerPerceptronClassifier().setLayers(Array(50, 200, 46)).setBlockSize(32)
    val pipeline2 = new Pipeline().setStages(Array(indexer, tokenizer, gloveVectorizer, classifier2))

    val model2 = pipeline2.fit(training)    
    // model2.save("bin/nlu2")

    val prediction = model2.transform(test)
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("f1")
    val score = evaluator.evaluate(prediction)
    print("test score = " + score)
    spark.stop()
  }
}
