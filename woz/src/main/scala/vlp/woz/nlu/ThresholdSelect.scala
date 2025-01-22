package vlp.woz.nlu

import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.dllib.utils.Shape

import scala.reflect.ClassTag

/**
 * A customized layer that operates on tensor to select index indices whose score are
 * larger than a threshold.
 *
 * @author phuonglh@gmail.com
 *
 * @param ev tensor numeric
 */
class ThresholdSelect[T: ClassTag](threshold: Float = 0.5f)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dimension = input.size.length
    val (values, indices) = input.max(dimension)
    val (v, i) = (values.asInstanceOf[Tensor[Float]].toArray(), indices.asInstanceOf[Tensor[Int]].toArray())
    // sort the score in descending order and filter elements greater than threshold
    val selectedIndices = v.zip(i).sortBy(_._1)(Ordering[Float].reverse).filter(_._1 >= threshold).map(_._2)
    // take 2 elements
    val choice = if (selectedIndices.isEmpty) Array(-1, -1) else {
      if (selectedIndices.length == 1) selectedIndices ++ Array(-1) else selectedIndices.take(2)
    }
    val result = Tensor(choice)
    output.resizeAs(result)
    result.cast[T](output)
    output.squeeze(dimension)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradOutput
  }
}

object ThresholdSelect {
  def apply[T: ClassTag](threshold: Float = 0.5f)(implicit ev: TensorNumeric[T]): ThresholdSelect[T] = new ThresholdSelect[T](threshold)
}

/**
 * Keras-style layer of the ThresholdSelect
 *
 * @param inputShape input shape
 * @param ev tensor numeric
 */

class ThresholdSelectLayer[T: ClassTag](val inputShape: Shape = null, threshold: Float = 0.5f)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[Int], T](KerasUtils.addBatch(inputShape)) with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[Int], T] = {
    val layer = ThresholdSelect(threshold)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[Int], T]]
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    Shape(input.slice(0, input.length - 1)) // don't take the last dimension
  }
}

object ThresholdSelectLayer {
  def apply[@specialized(Float, Double) T: ClassTag](inputShape: Shape = null, threshold: Float = 0.5f)(implicit ev: TensorNumeric[T]): ThresholdSelectLayer[T] = {
    new ThresholdSelectLayer[T](inputShape, threshold: Float)
  }
}