package vlp.woz.nlu

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.optim.{AccuracyResult, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

/**
 * phuonglh@gamil.com
 *
 * @param paddingValue a padding value in the target sequence, default is -1f.
 * @param ev
 *
 * Note: 1-based label index for token classification
 */
class TimeDistributedTop1Accuracy(paddingValue: Int = -1)(implicit ev: TensorNumeric[Float]) extends ValidationMethod[Float] {
  override def apply(output: Activity, target: Activity): ValidationResult = {
    var correct = 0
    var count = 0
    val _output = output.asInstanceOf[Tensor[Float]] // nDim = 3
    val _target = target.asInstanceOf[Tensor[Float]] // nDim = 2
    // split by batch size (dim = 1 of output and target)
    _output.split(1).zip(_target.split(1))
      .foreach { case (tensor, ys) =>
        // split by time slice (dim = 1 of tensor)
        val zs = tensor.split(1).map { t =>
          val (_, k) = t.max(1) // the label with max score
          k(Array(1)).toInt // k is a tensor => extract its value
        }
        val c = ys.toArray().map(_.toInt).filter(e => e != paddingValue).zip(zs)
          .map(p => if (p._1 == p._2) 1 else 0)
        correct += c.sum
        count += c.length
      }
    new AccuracyResult(correct, count)
  }
  override def format(): String = "TimeDistributedTop1Accuracy"
}
