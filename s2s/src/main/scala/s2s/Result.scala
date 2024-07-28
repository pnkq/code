package s2s

import org.apache.spark.sql.SparkSession
import scopt.OptionParser

case class ExperimentConfig(
  station: String,
  horizon: Int,
  lookBack: Int,
  layers: Int,
  hiddenSize: Int,
  epochs: Int,
  dropoutRate: Double,
  learningRate: Double,
  modelType: Int,
  heads: Int = 0,
  blocks: Int = 0,
  intermediateSize: Int = 0
)

case class Result(
  maeU: Array[Double],
  mseU: Array[Double],
  maeV: Array[Double],
  mseV: Array[Double],
  trainingTime: Long,
  config: ExperimentConfig
)


import org.apache.spark.sql.functions._

/**
 * phuonglh@gmail.com
 *
 */
object Result {

  private def analyzeLSTM(spark: SparkSession, config: Config): Unit = {
    val df = spark.read.json(s"dat/result-complex.jsonl")
    val station = df.filter(col("config.station") === config.station)
    val horizons = Array(7, 14, 21, 28)
    for (h <- horizons) {
      val stationH = station.filter(s"config.horizon == $h")
      var bf = stationH
      for (j <- 1 to h) {
        bf = bf.withColumn(s"h$j", format_number(element_at(col("mseV"), j), 6))
          .withColumn("n", lit(1)) // for counting
      }
      val aggF = (1 to h).map(j => s"h$j" -> "mean") :+ ("n" -> "count")
      val cf = bf.groupBy("config.horizon", "config.layers", "config.hiddenSize")
        .agg(aggF.head, aggF.tail:_*)
        .sort("horizon", "avg(h1)", "layers", "hiddenSize")
      var ef = cf
      for (j <- 1 to h) {
        ef = ef.withColumn(s"avg(h$j)", format_number(col(s"avg(h$j)"), 4))
      }
      ef.show()
      val n = ef.head.length
      val x = (3 until n-1).map { i => (i-2, ef.head.getAs[String](i)) }.mkString(" ")
      println(x) // for TikZ plot in the manuscript
    }
  }

  private def analyzeBERT(spark: SparkSession, config: Config): Unit = {
    val df = spark.read.json(s"dat/result-bert-${config.station}.jsonl")
    val station = df.filter(col("config.station") === config.station)
    val horizons = Array(7, 14, 21, 28)
    for (h <- horizons) {
      val stationH = station.filter(s"config.horizon == $h")
      var bf = stationH
      for (j <- 1 to h) {
        bf = bf.withColumn(s"h$j", format_number(element_at(col("mseV"), j), 6))
          .withColumn("n", lit(1)) // for counting
      }
      val aggF = (1 to h).map(j => s"h$j" -> "mean") :+ ("n" -> "count")
      val cf = bf.groupBy("config.horizon", "config.layers", "config.hiddenSize", "config.heads", "config.blocks")
        .agg(aggF.head, aggF.tail:_*)
        .sort("horizon", "avg(h1)", "layers", "hiddenSize", "heads", "blocks")
      var ef = cf
      for (j <- 1 to h) {
        ef = ef.withColumn(s"avg(h$j)", format_number(col(s"avg(h$j)"), 4))
      }
      ef.show()
      val n = ef.head.length
      val x = (5 until n-1).map { i => (i-4, ef.head.getAs[String](i)) }.mkString(" ")
      println(x) // for TikZ plot in the manuscript
    }

  }

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[Config](getClass.getName) {
      head(getClass.getName, "1.0")
      opt[String]('d', "data").action((x, conf) => conf.copy(data = x)).text("data type: simple/complex")
      opt[String]('s', "station").action((x, conf) => conf.copy(station = x)).text("station viet-tri/vinh-yen/...")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        val spark = SparkSession.builder().master("local[4]").appName(Result.getClass.getName).getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        println(f"Analyze results at station = ${config.station}: ")
        config.data match {
          case "bert" => analyzeBERT(spark, config) // LSTM+BERT
          case "lstm" => analyzeLSTM(spark, config) // LSTM on complex
        }
        spark.stop()

      case None => println("Invalid options!")
    }
  }
}