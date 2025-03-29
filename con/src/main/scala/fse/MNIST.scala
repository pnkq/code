package fse

import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

import java.awt.image.DataBufferByte
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.{Loss, Top1Accuracy, Top5Accuracy, Trigger}
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}

/**
 * (C) phuonglh@gmail.com
 *
 */
object MNIST {
  private val numCores = Runtime.getRuntime.availableProcessors()

  val decode = udf((bytes: Array[Byte]) => {
    val bufferImage = ImageIO.read(new ByteArrayInputStream(bytes))
    // an array of Byte of 784 integers, with negative values corrected
    val x = bufferImage.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
      .map(v => if (v < 0) v + 128 else v)
    Vectors.dense(x.map(_.toDouble))
  })

  private def ffn(train: DataFrame, test: DataFrame): Unit = {
    val classifier = new MultilayerPerceptronClassifier().setLayers(Array(784, 100, 10))
    val model = classifier.fit(train)
    val (uf, vf) = (model.transform(train), model.transform(test))
    val evaluator = new MulticlassClassificationEvaluator()
    val (a, b) = (evaluator.evaluate(uf), evaluator.evaluate(vf))
    println(s"training score = $a, test score = $b")
  }

  private def cnn: Sequential[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(1, 28, 28), inputShape = Shape(28 * 28)))
    model.add(Convolution2D(6, 5, 5, activation = "tanh").setName("conv1_5x5"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(12, 5, 5, activation = "tanh").setName("conv2_5x5"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100, activation = "tanh").setName("fc1"))
    model.add(Dense(10, activation = "softmax").setName("fc2"))
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

    // 1. FFN
//    ffn(train, test)

    // 2. CNN
    val inc = udf((i: Long) => i + 1d)
    val (uf, vf) = (
      train.withColumn("number", inc(col("label"))),
      test.withColumn("number", inc(col("label")))
    )
    val criterion = ClassNLLCriterion(sizeAverage = false, logProbAsInput = false)
    val estimator = NNEstimator(cnn, criterion, Array(28*28), Array(1))
    val trainingSummary = TrainSummary(appName = "cnn", logDir = "sum/cnn/")
    val validationSummary = ValidationSummary(appName = "cnn", logDir = "sum/cnn/")
    val batchSize = numCores * 4
    estimator.setBatchSize(batchSize).setOptimMethod(new Adam(2E-4)).setMaxEpoch(40)
      .setLabelCol("number")
      .setTrainSummary(trainingSummary).setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, vf, Array(new Top1Accuracy, new Top5Accuracy, new Loss), batchSize)
    estimator.fit(uf)

    spark.stop()
  }
}
