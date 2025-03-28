package fse

import com.intel.analytics.bigdl.dllib.keras.Model
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.{BCECriterion, ClassNLLCriterion, KLDCriterion, ParallelCriterion}
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.{Loss, Top1Accuracy, Trigger}
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import fse.IntentDetectionLSTM.{createModel, numCores}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.{col, udf}


/**
 * An implementation of Variational AutoEncoder (VAE) models on MNIST images.
 * (C) phuonglh@gmail.com
 */
object VAE {
  private val numCores = Runtime.getRuntime.availableProcessors()
  /**
   * Create an encoder graph for an image.
   *
   * @param nChannel
   * @param nCol
   * @param nRow
   * @param hiddenSize
   * @return a model.
   */
  private def createEncoder(nChannel: Int, nCol: Int, nRow: Int, hiddenSize: Int): Model[Float] = {
    val input = Input(inputShape = Shape(nChannel, nCol, nRow))
    val conv1 = Convolution2D(16, 5, 5, borderMode = "same", subsample = (2, 2)).inputs(input)
    val relu1 = LeakyReLU().inputs(conv1)
    val conv2 = Convolution2D(32, 5, 5, borderMode = "same", subsample = (2, 2)).inputs(relu1)
    val relu2 = LeakyReLU().inputs(conv2)
    val reshape = Flatten().inputs(relu2)
    val mean = Dense(hiddenSize).inputs(reshape)
    val logVar = Dense(hiddenSize).inputs(reshape)
    Model(input = input, output = Array(mean, logVar))
  }

  private def createDecoder(hiddenSize: Int): Model[Float] = {
    val input = Input(inputShape = Shape(hiddenSize))
    val dense = Dense(32 * 7 * 7, activation = "relu").inputs(input)
    val reshape = Reshape(targetShape = Array(32, 7, 7)).inputs(dense)
    val resize1 = ResizeBilinear(outputHeight = 14, outputWidth = 14).inputs(reshape)
    val conv1 = Convolution2D(16, 5, 5, subsample = (1, 1), activation = "relu", borderMode = "same").inputs(resize1)
    val resize2 = ResizeBilinear(outputHeight = 28, outputWidth = 28).inputs(conv1)
    val conv2 = Convolution2D(1, 5, 5, subsample = (1, 1), activation = "sigmoid", borderMode = "same").inputs(resize2)
    val output = Reshape(targetShape = Array(28*28)).inputs(conv2)
    Model(input = input, output = output)
  }

  private def vae(nChannel: Int, nCol: Int, nRow: Int, hiddenSize: Int): (Model[Float], Model[Float]) = {
    val input = Input(inputShape = Shape(nChannel * nCol * nRow))
    val reshape = Reshape(targetShape = Array(nChannel, nCol, nRow)).inputs(input)
    val encoderNode = createEncoder(nChannel, nCol, nRow, hiddenSize).inputs(reshape)
    val sampler = GaussianSampler().inputs(encoderNode)
    val decoder = createDecoder(hiddenSize)
    val decoderNode = decoder.inputs(sampler)
//    val model = Model(input = input, output = Array(encoderNode, decoderNode))
    val model = Model(input = input, output = decoderNode)
    (model, decoder)
  }

  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName(getClass.getName).setMaster("local[4]")
      .set("spark.executor.memory", "8g").set("spark.driver.memory", "8g")
    val sc = new SparkContext(conf)
    Engine.init
    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val (train, test) = (
      spark.read.parquet("dat/mnist/train-00000-of-00001.parquet").withColumn("features", fse.MNIST.decode(col("image.bytes"))),
      spark.read.parquet("dat/mnist/test-00000-of-00001.parquet").withColumn("features", fse.MNIST.decode(col("image.bytes")))
    )

    val (model, decoder) = vae(1, 28, 28, 10)
    model.summary()
    decoder.summary()
    val batchSize = numCores * 4
//    val criterion = ParallelCriterion()
//    criterion.add(KLDCriterion(), 1.0)
//    criterion.add(BCECriterion(sizeAverage = false), 1.0/batchSize)
    val criterion = BCECriterion(sizeAverage = false)

    val estimator = NNEstimator(model, criterion, Array(28*28), Array(28*28))
    val trainingSummary = TrainSummary(appName = "vae", logDir = "sum/vae/")
    val validationSummary = ValidationSummary(appName = "vae", logDir = "sum/vae/")
    estimator.setLabelCol("features")
      .setBatchSize(batchSize)
      .setOptimMethod(new Adam(2E-3))
      .setMaxEpoch(30)
      .setTrainSummary(trainingSummary)
    estimator.fit(test)

    spark.stop()
  }
}
