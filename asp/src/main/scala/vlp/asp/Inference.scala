package vlp.asp

object Inference {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName(getClass.getName)
      .master("local[*]")
      .config("spark.driver.host", "localhost")
      .config("spark.driver.memory", "8g")
      .config("spark.shuffle.blockTransferService", "nio")
      .getOrCreate()
    Engine.init

    spark.stop()

  }
}
