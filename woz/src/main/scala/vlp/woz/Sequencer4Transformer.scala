package vlp.woz

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataType


/**
  * A sequence vectorizer transforms a sequence of tokens into 2 sequences of indices for use
  * in a Transformer model. This transformer pads or truncate long sentence to a given `maxSequenceLength`.
  *
  * phuonglh@gmail.com
  */
class Sequencer4Transformer(val uid: String, val dictionary: Map[String, Int], val maxSequenceLength: Int, val padding: Int)
  extends UnaryTransformer[Seq[String], Vector, Sequencer4Transformer] with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[String, Int]]] = None
  var maxSeqLen: Int = -1
  var pad: Int = -1

  def this(dictionary: Map[String, Int], maxSequenceLength: Int, padding: Int) = {
    this(Identifiable.randomUID("seq4Trans"), dictionary, maxSequenceLength, padding)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(dictionary))
    this.maxSeqLen = maxSequenceLength
    this.pad = padding
  }

  override protected def createTransformFunc: Seq[String] => Vector = {
    def f(xs: Seq[String]): Vector = {
      val n = xs.size
      // token ids
      val tokens = xs.map(x => dictionaryBr.get.value.getOrElse(x, 0).toDouble).toArray
      // positions, start from 0 until xs.size
      val positions = Array.fill[Double](n)(0)
      for (j <- 0 until n)
        positions(j) = j
      // truncate or pad
      if (xs.size >= maxSeqLen) {
        Vectors.dense(tokens.take(maxSeqLen) ++ positions.take(maxSeqLen))
      } else {
        val a = tokens    ++ Array.fill[Double](maxSeqLen - xs.size)(pad)
        val c = positions ++ Array.fill[Double](maxSeqLen - xs.size)(pad)
        Vectors.dense(a ++ c)
      }
    }

    f
  }

  override protected def outputDataType: DataType = VectorType
}

object Sequencer4Transformer extends DefaultParamsReadable[Sequencer4Transformer] {
  override def load(path: String): Sequencer4Transformer = super.load(path)
}