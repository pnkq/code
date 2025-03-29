package fse

import com.intel.analytics.bigdl.dllib.feature.dataset.Sample
import com.intel.analytics.bigdl.dllib.keras.Model
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.{BCECriterion, KLDCriterion, ParallelCriterion}
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{col, udf}

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO


/**
 * An implementation of Variational AutoEncoder (VAE) models on MNIST images.
 * (C) phuonglh@gmail.com
 */
object VAE {
  private val numCores = Runtime.getRuntime.availableProcessors()

  /**
   * Create an encoder graph for an image.
   *
   * @param nChannel number of color channels (1, 3)
   * @param nCol number of rows (width)
   * @param nRow number of columns (height)
   * @param hiddenSize latent size of the encoder
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
    val model = Model(input = input, output = Array(encoderNode, decoderNode))
    (model, decoder)
  }

  private def export(x: Array[Float], path: String) = {
    println(x.mkString(", "))
    val g = x.map(v => Math.min(255, Math.max(0, v.toInt)))
    val image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)
    for (k <- g.indices) {
      val color = new Color(255-g(k), 255-g(k), 255-g(k))
      image.setRGB(k % 28, k / 28, color.getRGB)
    }
    ImageIO.write(image, "png", new File(path))
  }


  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName(getClass.getName).setMaster("local[4]")
      .set("spark.executor.memory", "8g").set("spark.driver.memory", "8g")
    val sc = new SparkContext(conf)
    Engine.init
    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val train = spark.read.parquet("dat/mnist/train-00000-of-00001.parquet")
      .withColumn("features", fse.MNIST.decode(col("image.bytes")))
    train.printSchema()
    train.select("label", "features").show(5, truncate = false)
    // export the first image
    val y = train.select("features").head.getAs[Vector](0).toArray.map(_.toFloat)
    export(y, "0.png")

    // threshold function on gray image pixels
    val sample = udf((x: Vector) => x.toArray.map(v => if (v > 128) 1f else 0f))
    val uf = train.withColumn("x", sample(col("features")))

    val hiddenSize = 2
    val (model, decoder) = vae(1, 28, 28, hiddenSize)
    model.summary()
    decoder.summary()

    val batchSize = numCores * 8
    val criterion = ParallelCriterion()
    criterion.add(KLDCriterion(), 1.0)
    criterion.add(BCECriterion(sizeAverage = false), 1.0/batchSize)
//    val criterion = BCECriterion(sizeAverage = false)

//    val estimator = NNEstimator(model, criterion, Array(28*28), Array(28*28))
//    val trainingSummary = TrainSummary(appName = "vae", logDir = "sum/")
//    estimator.setFeaturesCol("x").setLabelCol("x")
//      .setBatchSize(batchSize)
//      .setOptimMethod(new Adam(1E-3))
//      .setMaxEpoch(2)
//      .setTrainSummary(trainingSummary)
//    estimator.fit(uf)
//    model.saveModel("bin/vae.bigl", overWrite = true)
//    decoder.saveModel("bin/vae-dec.bigl", overWrite = true)

    val vf = uf.select("x").rdd.map(row => {
      val x = row.getSeq[Float](0).toArray
      val t = Tensor(x, Array(x.length))
      Sample[Float](featureTensors = Array(t), labelTensors = Array(t, t))
    })
    model.compile(optimizer = new Adam(1E-3), loss = criterion)
    model.setTensorBoard("./sum", "vae")
    model.fit(x = vf, batchSize = batchSize, nbEpoch = 1)
    model.saveModel("bin/vae.bigl", overWrite = true)
    decoder.saveModel("bin/vae-dec.bigl", overWrite = true)

    // generate some images from a multivariate normal distribution N(0, hiddenSize)
    val mean = Array.fill(hiddenSize)(0d)
    val covariance = Array.ofDim[Double](hiddenSize, hiddenSize)
    for (i <- 0 until hiddenSize) covariance(i)(i) = 1.0
    val mvn = new MultivariateNormalDistribution(mean, covariance)
    for (k <- 0 to 9) {
      val x = mvn.sample().map(_.toFloat)
      val y = decoder.forward(Tensor(x, Array(1, hiddenSize))).toTensor.squeeze().toArray().map(_ * 255)
      export(y, s"2/digits-$k.png")
    }

    spark.stop()
  }
}

