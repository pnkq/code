package s2s

import org.apache.spark.sql.SparkSession

case class ExperimentConfig(
  station: String,
  horizon: Int,
  lookBack: Int,
  layers: Int,
  hiddenSize: Int,
  epochs: Int,
  dropoutRate: Double,
  learningRate: Double,
  heads: Int = 0,
  blocks: Int = 0,
  intermediateSize: Int = 0,
  modelType: Int = 1
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

object Result {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[4]").appName(Result.getClass.getName).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val df = spark.read.json("dat/result-complex.jsonl")
    val stationName = if (args.length > 0) args(0) else "viet-tri"
    val station = df.filter(col("config.station") === stationName)
    val horizons = Array(5, 7, 10, 14)
    for (h <- horizons) {
      val stationH = station.filter(s"config.horizon == $h")
      var bf = stationH
      for (j <- 1 to h) {
        bf = bf.withColumn(s"h$j", format_number(element_at(col("mseV"), j), 6))
      }
//      bf.select((1 to h).map(j => col(s"h$j")): _*).show()
      val aggExps = (1 to h).map(j => s"h$j" -> "mean")
      val resultCols = (1 to h).map(j => s"avg(h$j)")
      val cf = bf.groupBy("config.lookBack", "config.horizon", "config.layers", "config.hiddenSize")
        .agg(aggExps.head, aggExps.tail:_*)
        .sort("lookBack", "horizon", "avg(h1)", "layers", "hiddenSize")
      var ef = cf
      for (j <- 1 to h) {
        ef = ef.withColumn(s"avg(h$j)", format_number(col(s"avg(h$j)"), 6))
      }
      ef.show()
    }

    spark.stop()
  }
}