package vlp.dep

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.linalg.{SQLDataTypes, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataType


/**
 * A Spark transformer which shift a vector by an offset. Padded values (-1d) are kept.
 * phuonglh@gmail.com
 * </p>
 * @param uid
 * @param offset
 */

class Shifter(override val uid: String, val offset: Int, padding: Float = -1f)
  extends UnaryTransformer[Vector, Vector, Shifter] with DefaultParamsWritable {

  def this(offset: Int, padding: Float) {
    this(Identifiable.randomUID("shifter"), offset, padding)
  }

  override protected def createTransformFunc: Vector => Vector = {
    def f(vector: Vector) = {
      Vectors.dense(vector.toArray.map(v => if (v != padding.toFloat) v + offset else padding))
    }

    f
  }
  override protected def outputDataType: DataType = SQLDataTypes.VectorType
}

object PretrainedEmbedding extends DefaultParamsReadable[Shifter] {
  override def load(path: String): Shifter = super.load(path)
}

