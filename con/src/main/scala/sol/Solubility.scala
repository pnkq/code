package sol

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{RegexTokenizer, StringIndexer}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.regression.GBTRegressionModel

/**
 * (C) phuonglh@gmail.com
 * 
 * An implementation of solubility prediction of protein using traditional ML models.
 *
 */
object Solubility {
  val f = udf((seq: String) => seq.toCharArray().map(_.toString()))

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local[4]").appName(getClass.getName).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val basePath = "dat/sol/"
    val paths = Array("eSol_train", "eSol_test", "S.cerevisiae_test").map(p => s"$basePath/$p.csv")
    val train = spark.read.option("header", value = true).option("inferSchema", true).csv(paths(0))
    val valid = spark.read.option("header", value = true).option("inferSchema", true).csv(paths(1))
    train.show(5)

    val uf = train.withColumn("tokens", f(col("sequence")))
    val vf = valid.withColumn("tokens", f(col("sequence")))
    uf.select("solubility", "tokens").show(5)

    val gbt = true
    val hashing = new HashingTF().setInputCol("tokens").setOutputCol("features").setNumFeatures(32)
    val regressor = if (!gbt) {
      new RandomForestRegressor().setFeaturesCol("features").setLabelCol("solubility")
    } else {
      new GBTRegressor().setFeaturesCol("features").setLabelCol("solubility")
    }
    val pipeline = new Pipeline().setStages(Array(hashing, regressor))

    val model = pipeline.fit(uf)
    val af = model.transform(uf)
    val bf = model.transform(vf)
    af.select("prediction", "solubility", "features").show(5)

    val evaluator = new RegressionEvaluator().setLabelCol("solubility").setPredictionCol("prediction").setMetricName("r2")
    var metric = evaluator.evaluate(bf)
    println(s" R^2 on test data = $metric")
    metric = evaluator.evaluate(af)
    println(s"R^2 on train data = $metric")

    val verbose = false
    if (verbose) {
      val regressor = if (!gbt) {
        model.stages(1).asInstanceOf[RandomForestRegressionModel] 
      } else {
        model.stages(1).asInstanceOf[GBTRegressionModel]
      }
      println(s"Learned regression model:\n ${regressor.toDebugString}")
    }

    spark.stop()
  }
}
