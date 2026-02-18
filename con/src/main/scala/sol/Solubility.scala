package sol

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{RegexTokenizer, StringIndexer}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{HashingTF, VectorAssembler}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.regression.GBTRegressionModel
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.Regressor
import org.apache.spark.ml.regression.LinearRegressionModel
import breeze.linalg.dim
import org.apache.spark.ml.feature.PCA

/**
 * (C) phuonglh@gmail.com
 * 
 * An implementation of solubility prediction of protein using traditional ML models.
 *
 */

object RegressorType extends Enumeration {
  val RANDOM_FOREST, GRADIENT_BOOSTED_TREE, LINEAR_REGRESSION = Value
}

object Solubility {

  def run(uf: DataFrame, vf: DataFrame, wf: DataFrame, esm: Boolean = false, dimensions: Int = 320,
    t: RegressorType.Value = RegressorType.RANDOM_FOREST, verbose: Boolean = false) = {
    val regressor = t match {
      case RegressorType.RANDOM_FOREST =>
        new RandomForestRegressor().setLabelCol("solubility").setNumTrees(128)
      case RegressorType.GRADIENT_BOOSTED_TREE =>
        new GBTRegressor().setLabelCol("solubility")
      case RegressorType.LINEAR_REGRESSION => 
        new LinearRegression().setLabelCol("solubility").setRegParam(1E-4) // avoid overfitting
      case _ =>
        new RandomForestRegressor().setLabelCol("solubility").setNumTrees(128)
    }
    val pipeline = if (!esm) {
      val hashing = new HashingTF().setInputCol("tokens").setOutputCol("features").setNumFeatures(32)
      new Pipeline().setStages(Array(hashing, regressor))
    } else {
      val columns = (0 until dimensions).map(j => s"dim_$j").toArray
      val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("x")
      val pca = new PCA().setInputCol("x").setOutputCol("features").setK(320)
      new Pipeline().setStages(Array(assembler, pca, regressor))
    }

    val model = pipeline.fit(uf)
    val af = model.transform(uf)
    val bf = model.transform(vf)
    val cf = model.transform(wf)
    af.select("prediction", "solubility", "features").show(5)

    if (verbose) {
      val n = model.stages.length - 1 // last stage, which is the regressor model
      t match {
        case RegressorType.RANDOM_FOREST => 
          val regressor = model.stages(n).asInstanceOf[RandomForestRegressionModel] 
          println(s"Learned regression model:\n ${regressor.toDebugString}")
        case RegressorType.GRADIENT_BOOSTED_TREE => 
          val regressor = model.stages(n).asInstanceOf[GBTRegressionModel]
          println(s"Learned regression model:\n ${regressor.toDebugString}")
        case RegressorType.LINEAR_REGRESSION => 
          val regressor = model.stages(n).asInstanceOf[LinearRegressionModel]
          println(s"Coefficients: ${regressor.coefficients} Intercept: ${regressor.intercept}")
          val trainingSummary = regressor.summary
          println(s"numIterations: ${trainingSummary.totalIterations}")
          println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
          trainingSummary.residuals.show()          
      }
    }
    val evaluator = new RegressionEvaluator().setLabelCol("solubility").setPredictionCol("prediction").setMetricName("r2")
    var metric = evaluator.evaluate(af)
    println(s"R^2 on train data = $metric")
    metric = evaluator.evaluate(bf)
    println(s"R^2 on valid data = $metric")
    metric = evaluator.evaluate(cf)
    println(s" R^2 on test data = $metric")
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local[*]").appName(getClass.getName).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val basePath = "dat/sol/"

    val esm: Boolean = true
    val dimensions = 1280 // 320
    val regressorType = RegressorType.RANDOM_FOREST
    println(s"esm = $esm, regressorType = $regressorType")

    val (uf, vf, wf) = if (!esm) {
      val paths = Array("eSol_train", "eSol_test", "S.cerevisiae_test").map(p => s"$basePath/$p.csv")
      val train = spark.read.option("header", value = true).option("inferSchema", true).csv(paths(0))
      val valid = spark.read.option("header", value = true).option("inferSchema", true).csv(paths(1))
      val test = spark.read.option("header", value = true).option("inferSchema", true).csv(paths(2))
      train.show(5)
      val f = udf((seq: String) => seq.toCharArray().map(_.toString()))
      val uf = train.withColumn("tokens", f(col("sequence")))
      val vf = valid.withColumn("tokens", f(col("sequence")))
      val wf = test.withColumn("tokens", f(col("sequence")))
      uf.select("solubility", "tokens").show(5)
      (uf, vf, wf)
    } else {
      val suffix = if (dimensions == 320) "8M" else "650M"
      val paths = Array("eSol_train", "eSol_test", "S.cerevisiae_test").map(p => s"$basePath/${p}_embeddings_${suffix}.parquet")
      val uf = spark.read.parquet(paths(0))
      val vf = spark.read.parquet(paths(1))
      val wf = spark.read.parquet(paths(2))
      (uf, vf, wf)
    }
    // 
    run(uf, vf, wf, esm, dimensions, regressorType, false)

    spark.stop()
  }
}

