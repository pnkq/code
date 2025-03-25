package fse

import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
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

  private val decode = udf((bytes: Array[Byte]) => {
    val bufferImage = ImageIO.read(new ByteArrayInputStream(bytes))
    // an array of Byte of 784 integers
    val x = bufferImage.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    Vectors.dense(x.map(_.toDouble))
  })

  private def ffn(train: DataFrame, test: DataFrame): Unit = {
    val classifier = new MultilayerPerceptronClassifier().setLayers(Array(784, 128, 10))
    val model = classifier.fit(train)
    val (uf, vf) = (model.transform(train), model.transform(test))
    val evaluator = new MulticlassClassificationEvaluator()
    val (a, b) = (evaluator.evaluate(uf), evaluator.evaluate(vf))
    println(s"training score = $a, test score = $b")
  }

  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName(getClass.getName).setMaster("local[4]")
      .set("spark.executor.memory", "8g").set("spark.driver.memory", "8g")
    val sc = new SparkContext(conf)
    Engine.init

    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val (train, test) = (
      spark.read.parquet("dat/mnist/train-00000-of-00001.parquet").withColumn("features", decode(col("image.bytes"))),
      spark.read.parquet("dat/mnist/test-00000-of-00001.parquet").withColumn("features", decode(col("image.bytes")))
    )

    ffn(train, test)

    spark.stop()
  }
}
