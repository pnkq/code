package vlp.dep

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.ArrayType


/**
  * A sequence vectorizer transforms a sequence of tokens into a sequence of vectors
  * using a dictionary. This transformer pads or truncate long sentence to a given `maxSequenceLength`.
  * If the dictionary does not contain a token, it returns a zero vector.
  *
  * <p>
  * phuonglh@gmail.com, August 19, 2024.
  */
class SequencerX(val uid: String, val dictionary: Map[String, Seq[Double]], val maxSequenceLength: Int, val defaultSize: Int) 
  extends UnaryTransformer[Seq[String], Seq[Vector], SequencerX] with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[String, Seq[Double]]]] = None
  var maxSeqLen: Int = -1

  def this(dictionary: Map[String, Seq[Double]], maxSequenceLength: Int, defaultSize: Int = 3) = {
    this(Identifiable.randomUID("seq"), dictionary, maxSequenceLength, defaultSize)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(dictionary))
    this.maxSeqLen = maxSequenceLength
  }

  override protected def createTransformFunc: Seq[String] => Seq[Vector] = {
    def f(xs: Seq[String]): Seq[Vector] = {
      val ys = xs.map(x => dictionaryBr.get.value.getOrElse(x, List.fill[Double](defaultSize)(0d))).map(a => Vectors.dense(a.toArray))
      // truncate or pad
      if (ys.size >= maxSeqLen) {
        ys.take(maxSeqLen)
      } else {
        ys ++ Array.fill[Vector](maxSeqLen - ys.size)(Vectors.zeros(defaultSize)) 
      }
    }

    f
  }

  override protected def outputDataType: DataType = ArrayType(VectorType, containsNull = false)
}

object SequencerX extends DefaultParamsReadable[SequencerX] {
  override def load(path: String): SequencerX = super.load(path)
}