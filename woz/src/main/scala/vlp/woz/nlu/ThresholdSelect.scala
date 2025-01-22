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
    val (v, i) = (values.asInstanceOf[Tensor[Float]], indices.asInstanceOf[Tensor[Float]])
    val result = v.squeeze(dimension).map(i.squeeze(dimension), (a, b) => a + b) // 42.28447 => [index = 42, value = 0.28847]
    output.resizeAs(result)
    result.cast[T](output)
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
    Shape(input.slice(0, input.length - 1))
  }
}

object ThresholdSelectLayer {
  def apply[@specialized(Float, Double) T: ClassTag](inputShape: Shape = null, threshold: Float = 0.5f)(implicit ev: TensorNumeric[T]): ThresholdSelectLayer[T] = {
    new ThresholdSelectLayer[T](inputShape, threshold: Float)
  }
}