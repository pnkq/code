package vlp.nlu

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.{Pipeline, PipelineModel}

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.embeddings.{BertSentenceEmbeddings, RoBertaSentenceEmbeddings}
import com.johnsnowlabs.nlp.{DocumentAssembler, EmbeddingsFinisher}

import scala.io.Source
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


object PretrainedNLU {

  def train(trainingDF: DataFrame, modelType: String="b"): PipelineModel = {
    val document = new DocumentAssembler().setInputCol("answer_normalised").setOutputCol("document")
    val embeddings = modelType match {
      case "b" => BertSentenceEmbeddings.pretrained().setInputCols("document").setOutputCol("embeddings")
      case "r" => RoBertaSentenceEmbeddings.pretrained().setInputCols("document").setOutputCol("embeddings")
    }
    val finisher = new EmbeddingsFinisher().setInputCols("embeddings").setOutputCols("vectors").setOutputAsVector(true)
    val selector = new Selector().setInputCol("vectors").setOutputCol("features")
    val indexer = new StringIndexer().setInputCol("intent").setOutputCol("label")
    val classifier = new LogisticRegression().setRegParam(1E-5)
    val pipeline = new Pipeline().setStages(Array(document, embeddings, finisher, selector, indexer, classifier))
    val model = pipeline.fit(trainingDF)
    model.write.overwrite.save("bin/nlu")
    model
  }


  def main(args: Array[String]): Unit = {
    // create Spark context
    val conf = new SparkConf().setAppName(getClass.getName).setMaster("local[*]")
      .set("spark.executor.cores", "4")
      .set("spark.cores.max", "4")
      .set("spark.executor.memory", "8g")
      .set("spark.driver.memory", "8g")
    val sc = new SparkContext(conf)
    sc.setLogLevel("INFO")

    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()

    // read and split data into train/dev.
    val df = spark.read.options(Map("delimiter" -> ";", "header" -> "true"))
      .csv("dat/nlu/nlu.csv.gz").na.drop().select("intent", "answer_normalised").sample(0.1)
    df.show(20, false)
    val Array(trainingDF, testDF) = df.randomSplit(Array(0.8, 0.2), seed = 1234L)
    println(s"Number of training samples = ${trainingDF.count}")
    println(s"    Number of test samples = ${testDF.count}")

    // train a pipeline 
    val model = train(trainingDF, "b")
    val prediction = model.transform(testDF)
    prediction.printSchema()
    println(prediction.select("vectors").head())
    
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("f1")
    val score = evaluator.evaluate(prediction)
    print("test score = " + score)
    
    spark.stop()
  }

}