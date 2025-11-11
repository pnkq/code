package vlp.asp

import org.apache.spark.sql.SparkSession
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.keras.models.Models
import scala.util.Try


object Inference {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName(getClass.getName)
      .master("local[*]")
      .config("spark.driver.host", "localhost")
      .config("spark.driver.memory", "8g")
      .config("spark.shuffle.blockTransferService", "nio")
      .getOrCreate()
    Engine.init

    val modelPath = "bin/eng.bigdl.osx"
    val model = Try(Models.loadModel[Float](modelPath)).getOrElse {
      println("First load failed, retrying...")
      Models.loadModel[Float](modelPath)
    }
    model.summary()


    spark.stop()

  }
}
