package com.intel.analytics.bigdl.phuonglh

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{SizeAverageStatus, TensorCriterion}

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.dllib.nn.ops.Slice

/**
  * This loss function is not used. We use another trick for the original ClassNLLCriterion.
  * 
  * phuonglh@gmail.com
  *
  * @param firstLoss
  * @param secondLoss
  * @param m
  * @param numOffsets
  * @param numLabels
  * @param ev
  */
class JointClassNLLCriterion[@specialized(Float, Double) T: ClassTag](
  firstLoss: ClassNLLCriterion[T], secondLoss: ClassNLLCriterion[T], m: Int, numOffsets: Int, numLabels: Int)(implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  sizeAverageStatus = SizeAverageStatus.True

  val sliceX1 = new Slice[Float](begin = Array(0, 0), size = Array(m, numOffsets))
  val sliceX2 = new Slice[Float](begin = Array(0, numOffsets), size = Array(m, numLabels))
    
  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    print(input)
    print(target)
    val output1 = sliceX1.forward(input)
    val output2 = sliceX2.forward(input)
    ev.plus(firstLoss.updateOutput(output1, target), secondLoss.updateOutput(output2, target))
  }
  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    val output1 = sliceX1.forward(input)
    val output2 = sliceX2.forward(input)
    val gradInput1 = updateGradInput(output1, target).asInstanceOf[Tensor[T]]
    val gradInput2 = updateGradInput(output2, target).asInstanceOf[Tensor[T]]
    gradInput.resizeAs(input)
    gradInput.zero()
    // update the gradInput
    val Array(b, r, c) = gradInput1.size() // should be the same as gradInput2.size()
    for (i <- 1 to b) 
      for (j <- 1 to r) {
        for (k <- 1 to c) {
          gradInput.setValue(i, j, k, ev.plus(gradInput1.valueAt(i, j, k), gradInput2.valueAt(i, j, k)))
        }
      }
    gradInput
  }
}
