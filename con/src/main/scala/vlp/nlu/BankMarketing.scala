package vlp.nlu

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object BankMarketing {
  def main(args: Array[String]): Unit = {
    // 0. create a Spark session
    val spark = SparkSession.builder().master("local[4]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    // 1. read df
    val df = spark.read.options(Map("header" -> "true", "delimiter" -> ";", "inferSchema" -> "true"))
      .csv("/home/phuonglh/Downloads/bank-marketing/full.csv")
    df.show(5, false)
    
    // // 2. create a preprocessing pipeline
    // val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("mIdx")
    // val maritalEncoder = new OneHotEncoder().setInputCol("mIdx").setOutputCol("mVec")
    // val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("eIdx")
    // val educationEncoder = new OneHotEncoder().setInputCol("eIdx").setOutputCol("eVec")
    // val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")

    // val assembler = new VectorAssembler().setInputCols(Array("age", "mVec", "eVec", "duration")).setOutputCol("features")
    // val classifier = new LogisticRegression()

    // val pipeline = new Pipeline().setStages(Array(maritalIndexer, maritalEncoder, educationIndexer, 
    //   educationEncoder, labelIndexer, assembler, classifier))

    // val Array(trainingDf, testDf) = df.randomSplit(Array(0.8, 0.2), seed=1234L)  
    // val model = pipeline.fit(trainingDf)

    // // model.save("bin/bm")
    // val ef = model.transform(testDf)
    // val output = ef.select("label", "prediction", "rawPrediction")
    // print("test size = ", output.count())

    // val evaluator = new BinaryClassificationEvaluator()
    // val score = evaluator.evaluate(output)
    // println(s"Test AUC score = $score.")

    spark.stop()
  }
}
