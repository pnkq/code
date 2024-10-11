package vlp.w2v

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.linalg.{SQLDataTypes, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataType


/**
 * A Spark transformer which computes text representation using pretrained embedding vectors.
 * phuonglh@gmail.com
 * </p>
 * @param uid
 * @param dictionary
 */

class PretrainedEmbedding(override val uid: String, val dictionary: Map[String, Seq[Double]], val dimension: Int)
  extends UnaryTransformer[Seq[String], Vector, PretrainedEmbedding] with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[String, Seq[Double]]]] = None
  val zero: Seq[Double] = Array.fill(dimension)(0d)

  def this(dictionary: Map[String, Seq[Double]], dimension: Int) = {
    this(Identifiable.randomUID("embedding"), dictionary, dimension)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(dictionary))
  }

  override protected def createTransformFunc: Seq[String] => Vector = {
    def avg(tokens: Seq[String]): Vector = {
      val dict = dictionaryBr.get.value
      val vs = tokens.map { token => dict.getOrElse(token, zero) }
      val s = vs.reduce { (u, v) => u.zip(v).map { p => p._1 + p._2 }}
      Vectors.dense(s.toArray.map(v => v/tokens.size))
    }
    avg(_)
  }
  override protected def outputDataType: DataType = SQLDataTypes.VectorType
}

object PretrainedEmbedding extends DefaultParamsReadable[PretrainedEmbedding] {
  override def load(path: String): PretrainedEmbedding = super.load(path)
}

