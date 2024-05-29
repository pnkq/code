package s2s

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object DataReader {

  /**
   * Reads a CSV file containing rainfall data of all stations.
   * @param spark a Spark session
   * @param path a path
   * @param station a station name (vinh-yen, tuan-giao, etc)
   * @return extract data specific to a station.
   */
  def readSimple(spark: SparkSession, path: String, station: String): DataFrame = {
    val df = spark.read.options(Map("delimiter" -> "\t", "inferSchema" -> "true")).csv(path)
    val stationMap = Map("muong-te" -> 0, "tuan-giao" -> 5, "son-la" -> 9, "sa-pa" -> 17, "ha-giang" -> 22, "viet-tri" -> 35, "vinh-yen" -> 36)
    val stationCol = s"_c${stationMap(station) + 3}"
    val ef = df.select("_c0", "_c1", "_c2", stationCol).toDF("year", "month", "dayOfMonth", stationCol)
    val prependZero = udf((x: Int) => if (x < 10) "0" + x.toString else x.toString)
    ef.withColumn("yearSt", col("year").cast("string"))
      .withColumn("monthSt", prependZero(col("month")))
      .withColumn("daySt", prependZero(col("dayOfMonth")))
      .withColumn("dateSt", concat_ws("/", col("yearSt"), col("monthSt"), col("daySt")))
      .withColumn("date", to_date(col("dateSt"), "yyy/MM/dd"))
      .withColumnRenamed(stationCol, "y")
  }

  /**
   * Reads a complex CSV file containing more than a hundred of columns. The label (rainfall) column is named "y".
   * Should remove data of year >= 2020 (many missing re-analysis data).
   *
   * @param spark spark session
   * @param path  path to the CSV file
   * @return a data frame and an array of date columns
   */
  def readComplex(spark: SparkSession, path: String): DataFrame = {
    val cf = spark.read.options(Map("inferSchema" -> "true", "header" -> "true")).csv(path)
    val selectedColNames = cf.schema.fieldNames.filter(name => name.contains("extra"))
    val df = cf.select((Array("Date", "y") ++ selectedColNames).map(name => col(name)): _*)
    val ef = df.withColumn("date", to_date(col("Date"), "yyy-MM-dd"))
      .withColumn("year", year(col("date")))
      .withColumn("month", month(col("date")))
      .withColumn("dayOfMonth", dayofmonth(col("date")))
      .withColumn("dayOfYear", dayofyear(col("date")))
    ef.filter("year < 2020")
  }

  /**
   * Reads the data from a cluster (simple format).
   * @param spark spark session
   * @param path path to a CSV file
   */
  def readClusterSimple(spark: SparkSession, path: String): DataFrame = {
    val cf = spark.read.options(Map("inferSchema" -> "true", "header" -> "true")).csv(path)
    val df = cf.select("Date", "y_mean")
    df.withColumn("date", to_date(col("Date"), "yyy-MM-dd"))
      .withColumn("year", year(col("date")))
      .withColumn("month", month(col("date")))
      .withColumn("dayOfMonth", dayofmonth(col("date")))
      .withColumn("dayOfYear", dayofyear(col("date")))
      .withColumnRenamed("y_mean", "y")
  }

  /**
   * Reads a complex CSV file containing more than a hundred of columns. The label (rainfall) column is named "y".
   * Should remove data of year >= 2020 (many missing re-analysis data).
   *
   * @param spark spark session
   * @param path  path to the CSV file
   * @return a data frame and an array of date columns
   */
  def readClusterComplex(spark: SparkSession, path: String): DataFrame = {
    val df = spark.read.options(Map("inferSchema" -> "true", "header" -> "true")).csv(path)
    val ef = df.withColumn("date", to_date(col("Date"), "yyy-MM-dd"))
      .withColumn("year", year(col("date")))
      .withColumn("month", month(col("date")))
      .withColumn("dayOfMonth", dayofmonth(col("date")))
      .withColumn("dayOfYear", dayofyear(col("date")))
    ef.filter("year < 2020")
  }

}
