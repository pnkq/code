package com.intel.analytics.bigdl.phuonglh

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{SizeAverageStatus, TensorCriterion}

import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.keras.layers.SelectTable
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.dllib.keras.Sequential

class JointClassNLLCriterion[@specialized(Float, Double) T: ClassTag](
  firstLoss: ClassNLLCriterion[T], secondLoss: ClassNLLCriterion[T])(implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  val sequentialX1 = Sequential[Float]()
  val sequentialX2 = Sequential[Float]()
  val sequentialY1 = Sequential[Float]()
  val sequentialY2 = Sequential[Float]()
  
  sizeAverageStatus = SizeAverageStatus.True

  def this(firstLoss: ClassNLLCriterion[T], secondLoss: ClassNLLCriterion[T], m: Int, n: Int)(implicit ev: TensorNumeric[T]) = {
    this(firstLoss, secondLoss)
    // remove a duplicate half of the input: split into two sequences, each of m time steps
    val splitXa = SplitTensor[Float](1, 2, inputShape=Shape(2*m, 2*n))
    val selectXa = SelectTable[Float](0)
    // b version is a copy of a
    val splitXb = SplitTensor[Float](1, 2, inputShape=Shape(2*m, 2*n))
    val selectXb = SelectTable[Float](0)
    // split input (the first half) of shape m x (2n)
    val splitX1 = SplitTensor[Float](1, 2)
    val selectX1 = SelectTable[Float](0)
    sequentialX1.add(splitXa).add(selectXa).add(splitX1).add(selectX1)

    val splitX2 = SplitTensor[Float](1, 2)
    val selectX2 = SelectTable[Float](1)
    val squeezeX2 = Squeeze[Float](1)
    sequentialX2.add(splitXb).add(selectXb).add(splitX2).add(selectX2)
    // split target of this loss
    val splitY1 = SplitTensor[Float](0, 2, inputShape=Shape(2*m))
    val selectY1 = SelectTable[Float](0)
    sequentialY1.add(splitY1).add(selectY1)

    val splitY2 = SplitTensor[Float](0, 2, inputShape=Shape(2*m))
    val selectY2 = SelectTable[Float](1)
    sequentialY2.add(splitY2).add(selectY2)
  }
    
  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    val output1 = sequentialX1.forward(input).asInstanceOf[Tensor[T]]
    val output2 = sequentialX2.forward(input).asInstanceOf[Tensor[T]]
    val target1 = sequentialY1.forward(target).asInstanceOf[Tensor[T]]
    val target2 = sequentialY2.forward(target).asInstanceOf[Tensor[T]]
    ev.plus(firstLoss.updateOutput(output1, target1), secondLoss.updateOutput(output2, target2))
  }
  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    val output1 = sequentialX1.forward(input).asInstanceOf[Tensor[T]]
    val output2 = sequentialX2.forward(input).asInstanceOf[Tensor[T]]
    val target1 = sequentialY1.forward(target).asInstanceOf[Tensor[T]]
    val target2 = sequentialY2.forward(target).asInstanceOf[Tensor[T]]
    val gradInput1 = updateGradInput(output1, target1).asInstanceOf[Tensor[T]]
    val gradInput2 = updateGradInput(output2, target2).asInstanceOf[Tensor[T]]
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
