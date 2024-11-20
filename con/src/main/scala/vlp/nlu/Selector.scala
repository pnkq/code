package vlp.nlu

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, FloatType}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType


/**
  * Select the first vector of a sequence of vectors.
  *
  * phuonglh@gmail.com
  */
class Selector(val uid: String)
  extends UnaryTransformer[Seq[Vector], Vector, Selector] with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("selector"))
  }

  override protected def createTransformFunc: Seq[Vector] => Vector = {
    def f(xs: Seq[Vector]): Vector = xs.head

    f(_)
  }

  override protected def outputDataType: DataType = VectorType
}

object Selector extends DefaultParamsReadable[Selector] {
  override def load(path: String): Selector = super.load(path)
}