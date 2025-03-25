package fse

import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

import java.awt.image.DataBufferByte
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

/**
 * (C) phuonglh@gmail.com
 */
object MNIST {
  private val numCores = Runtime.getRuntime.availableProcessors()

  val decode = udf((png: Array[Byte]) => {
    val bufferImage = ImageIO.read(new ByteArrayInputStream(png))
    bufferImage.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData // an array of Byte of 784 integers
  })
  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName(getClass.getName).setMaster("local[4]")
      .set("spark.executor.memory", "8g").set("spark.driver.memory", "8g")
    val sc = new SparkContext(conf)
    Engine.init

    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    spark.stop()
  }
}
