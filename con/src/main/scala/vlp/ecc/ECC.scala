package vlp.ecc

import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization
import org.apache.spark.sql.types.DoubleType
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.embeddings.{BertSentenceEmbeddings, RoBertaSentenceEmbeddings}
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.{Pipeline, PipelineModel}
import scopt.OptionParser
import org.apache.spark.ml.feature.{RegexTokenizer, CountVectorizer, Normalizer, VectorAssembler}
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import com.johnsnowlabs.nlp.annotator.DeBertaForSequenceClassification

/**
 * Phuong Le-Hong, <phuonglh@gmail.com>
 * 
 * December 2024.
 *
 */
object ECC {
  /**
    * Remove a prefix string (before the first ':'), for example "claim:" or "premises:".
    *
    * @param text a given text
    * @return a text with prefix removed.
    */
  def removePrefix(text: String): String = {
    val j = text.indexOf(':')
    if (j < 0) text else text.substring(j+1).trim
  }

  val f = udf((text: String) => removePrefix(text))

  implicit val formats: Formats = Serialization.formats(NoTypeHints)

  def extractPremises(premises: String): List[String] = {
    val text = premises.replaceAll(", '", ", \"")
      .replaceAll("', ", "\", ")
      .replaceAll("""\['""", """\["""")
      .replaceAll("""'\]""", """"\]""")

    parse(text).extract[List[String]]
  }

  val g = udf((text: String) => extractPremises(text))

  def preprocess(df: DataFrame): DataFrame = {
    val ef = df.withColumn("claim", f(col("claim_text"))).withColumn("premises", f(col("premise_texts")))
    val gf = ef.withColumn("xs", g(col("premises")))
    val hf = gf.select("sId", "claim", "year", "quarter", "label", "xs")
      .withColumn("premise", explode(col("xs")))
      .withColumn("target", col("label").cast(DoubleType))
      .withColumn("id", monotonically_increasing_id)
    hf
  }

  def preprocessJSL(df: DataFrame, config: ConfigECC, columnName: String): PipelineModel = {
    val document = new DocumentAssembler().setInputCol(columnName).setOutputCol("document")
    val embeddings = config.modelType match {
      case "b" => BertSentenceEmbeddings.pretrained().setInputCols("document").setOutputCol("embeddings").setMaxSentenceLength(256).setCaseSensitive(true)
      case "d" => DeBertaForSequenceClassification.pretrained().setInputCols("document").setOutputCol("embeddings").setMaxSentenceLength(256).setCaseSensitive(true)
      case "f" => BertSentenceEmbeddings.pretrained("sbert_setfit_finetuned_financial_text_classification","en").setInputCols("document").setOutputCol("embeddings").setMaxSentenceLength(256).setCaseSensitive(true)
      case "r" => RoBertaSentenceEmbeddings.pretrained().setInputCols("document").setOutputCol("embeddings").setMaxSentenceLength(256).setCaseSensitive(true)
    }
    val finisher = new EmbeddingsFinisher().setInputCols("embeddings").setOutputCols(s"${columnName}Vec").setOutputAsVector(true)
    val pipeline = new Pipeline().setStages(Array(document, embeddings, finisher))
    pipeline.fit(df)
  }

  val quarterMap = Map("Q1" -> 0, "Q2" -> 1, "Q3" -> 2, "Q4" -> 3)

  val h = udf((quarter: String) => quarterMap(quarter))

  val durationKeywords = Array("quarter", "year", "month", "Q1", "Q2", "Q3", "Q4")
  val d = udf((text: String) => {
    val vector = Array.fill[Double](durationKeywords.size)(0d)
    for (i <- 0 until durationKeywords.size) {
      if (text.contains(durationKeywords(i))) vector(i) = 1 else vector(i) = 0
    }
    Vectors.dense(vector)
  })

  def mlp(df: DataFrame, discreteFeatures: Boolean = false): PipelineModel = {
    val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern("""[\s,'.’]""")
    val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("vector").setMinDF(2)
    val normalizer = new Normalizer().setInputCol("vector").setOutputCol("v")
    // d: duration keyword vector; p: premise pretrained vector; c: claim pretrained vector
    // v: discrete count vector for text = premise :: claim
    val columns = if (discreteFeatures) Array("v", "p", "c", "d") else Array("p", "c", "d")
    val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")
    val classifier = new LogisticRegression().setLabelCol("target")
    val stages = if (discreteFeatures) Array(tokenizer, vectorizer, normalizer, assembler, classifier) else Array(assembler, classifier)
    val pipeline = new Pipeline().setStages(stages)
    pipeline.fit(df)
  }

  // majority voting from a list of predicted labels for a given sample Id ("sId")
  val vote = udf((a: Array[Double]) => {
    val g = a.groupBy(identity).mapValues(_.size)
    g.toList.maxBy(_._2)._1
  }) 


  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[ConfigECC](getClass.getName) {
      head(getClass.getName, "1.0")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either {eval, train, predict}")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x)).text("learning rate, default value is 1E-5")
      opt[String]('d', "trainPath").action((x, conf) => conf.copy(trainPath = x)).text("training data directory")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model folder, default is 'bin/'")
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path")
      opt[String]('s', "scorePath").action((x, conf) => conf.copy(scorePath = x)).text("score path")
      opt[Boolean]('c', "discreteFeatures").action((_, conf) => conf.copy(discreteFeatures = true)).text("using count features")
    }
    opts.parse(args, ConfigECC()) match {
      case Some(config) =>
        val spark = SparkSession.builder().config("spark.driver.memory", "8g").master("local[4]").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        config.mode match {
          case "prepare" =>  
            val paths = Array("dat/ecc/ECC-train.tsv", "dat/ecc/ECC-val.tsv")
            val Array(df, dfV) = paths.map{ path => 
              val af = spark.read.options(Map("delimiter" -> "\t", "header" -> "true")).csv(path)
              preprocess(af)
            }
            df.printSchema()
            println(s"Number of (train, valid) samples = (${df.count}, ${dfV.count}).")
            // JSL preprocessors 
            val preprocessorClaim = preprocessJSL(df, config, "claim")
            val preprocessorPremise = preprocessJSL(df, config, "premise")
            val (cf, cfV) = (preprocessorClaim.transform(df), preprocessorClaim.transform(dfV))
            val (pf, pfV) = (preprocessorPremise.transform(df), preprocessorPremise.transform(dfV))
            val firstCols = Seq("sId", "claim", "year", "quarter", "target", "claimVec")
            val secondCols = Seq("premise", "premiseVec")
            val ef = cf.select("id", firstCols: _*).join(pf.select("id", secondCols: _*), "id")
            val efV = cfV.select("id", firstCols: _*).join(pfV.select("id", secondCols: _*), "id")
            ef.printSchema()
            ef.show()
            // save the two data frames for fast loading later
            ef.write.mode("overwrite").parquet(s"${config.trainPath}-${config.modelType}")
            efV.write.mode("overwrite").parquet(s"${config.validPath}-${config.modelType}")
          case "train" => 
            val ef = spark.read.parquet(s"${config.trainPath}-${config.modelType}")
            val efV = spark.read.parquet(s"${config.validPath}-${config.modelType}")
            println(s"Number of (train, valid) samples = (${ef.count}, ${efV.count}).")
            ef.show()

            val df = ef.withColumn("p", explode(col("premiseVec"))).withColumn("c", explode(col("claimVec")))
              .withColumn("q", h(col("quarter"))).withColumn("text", concat_ws(" ", col("claim"), col("premise")))
              .withColumn("d", d(col("text")))
            val dfV = efV.withColumn("p", explode(col("premiseVec"))).withColumn("c", explode(col("claimVec")))
              .withColumn("q", h(col("quarter"))).withColumn("text", concat_ws(" ", col("claim"), col("premise")))
              .withColumn("d", d(col("text")))
            val model = mlp(df, config.discreteFeatures)
            val (ff, ffV) = (model.transform(df), model.transform(dfV))
            ff.show()
            val evaluator = new MulticlassClassificationEvaluator().setLabelCol("target")
            var score = evaluator.evaluate(ff)
            println(s"  training score = ${score}.")
            score = evaluator.evaluate(ffV)
            println(s"validation score = ${score}.")

            val (gf, gfV) = (
              ff.groupBy("sId").agg(collect_list("target").as("ys"), collect_list("prediction").as("zs"))
                .withColumn("label", vote(col("ys"))).withColumn("prediction", vote(col("zs"))),
              ffV.groupBy("sId").agg(collect_list("target").as("ys"), collect_list("prediction").as("zs"))
              .withColumn("label", vote(col("ys"))).withColumn("prediction", vote(col("zs"))),
            )
            println(s"Number of original (train, valid) samples = (${gf.count}, ${gfV.count}).")
            val finalEvaluator = new MulticlassClassificationEvaluator()
            var finalScore = finalEvaluator.evaluate(gf)
            println(s"  final training score = ${finalScore}.")
            finalScore = finalEvaluator.evaluate(gfV)
            println(s"final validation score = ${finalScore}.")
        }
        spark.stop()
      case None => 
    }
  }
}
