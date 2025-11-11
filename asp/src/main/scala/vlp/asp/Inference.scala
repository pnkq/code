package vlp.asp

import org.apache.spark.sql.SparkSession
import com.intel.analytics.bigdl.dllib.utils.Engine

object Inference {
  def main(args: Array[String]): Unit = {
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
