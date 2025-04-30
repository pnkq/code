package fse

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession

object BankMarketing {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val path = "/Users/phuonglh/Downloads/bank-full.csv"
    val df = spark.read.options(Map("header" -> "true", "delimiter" -> ";", "inferSchema" -> "true")).csv(path)
    df.show(10)

    val indexer = new StringIndexer().setInputCol("y").setOutputCol("label")
    val indexerEdu = new StringIndexer().setInputCol("education").setOutputCol("eduIdx")
    val encoderEdu = new OneHotEncoder().setInputCol("eduIdx").setOutputCol("eduVec")
    val indexerJob = new StringIndexer().setInputCol("job").setOutputCol("jobIdx")
    val encoderJob = new OneHotEncoder().setInputCol("jobIdx").setOutputCol("jobVec")
    val assembler = new VectorAssembler().setInputCols(Array("age", "balance", "duration", "eduVec", "jobVec")).setOutputCol("features")
//    val classifier = new LogisticRegression()
    val classifier = new MultilayerPerceptronClassifier().setLayers(Array(17, 2, 2))

    val pipeline = new Pipeline().setStages(Array(indexer, indexerEdu, encoderEdu, indexerJob, encoderJob, assembler, classifier))
    val Array(training, test) = df.randomSplit(Array(0.8, 0.2), seed=1403)
    val model = pipeline.fit(training)

    val (uf, vf) = (model.transform(training), model.transform(test))
    uf.select("features", "label").show(10)

    val evaluator = new BinaryClassificationEvaluator()//.setMetricName("areaUnderROC")
    val trainScore = evaluator.evaluate(uf)
    val testScore = evaluator.evaluate(vf)

    println("trainingScore = " + trainScore)
    println("testScore = " + testScore)

    spark.stop()
  }
}
