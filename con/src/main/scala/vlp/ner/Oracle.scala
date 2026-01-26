package vlp.ner

import scala.collection.mutable.{ListBuffer, Stack, Queue}
import java.util.concurrent.atomic.AtomicInteger


/**
  * Created by phuonglh on January 9, 2026.
  * 
  * A parsing context contains a stack, a queue, an output queue and bag-of-features 
  *  (space-separated feature strings) and a transition label.
  * 
  */
case class Context(id: Int, bof: String, transition: String, stack: Stack[Int], queue: Queue[Int], words: Seq[String], pastTransitions: Seq[String]) {
  override def toString: String = {
    val sb = new StringBuilder()
    sb.append('(')
    sb.append(id)
    // sb.append(',')
    // sb.append(bof)
    sb.append(',')
    sb.append(stack.clone().reverse)
    sb.append(',')
    sb.append(queue)
    sb.append(')')
    sb.append(" => ")
    sb.append(transition)
    sb.append(',')
    sb.append(pastTransitions)
    sb.toString()
  }
}

case class Sample(words: Seq[String], spans: Map[(Int, Int), String]) 


/**
  * Oracle is a key component in training transition-based NER. It is used to derive an
  * optimal transition sequence from a named entity tagged sequence.
  *
  */
object Oracle {
  val counter = new AtomicInteger(0)

  def decode(sample: Sample): List[Context] = decode(sample.words, sample.spans)

  def decode(words: Seq[String], spans: Map[(Int, Int), String]): List[Context] = {
    val n = words.size
    val queue = Queue[Int]()
    for (i <- 0 until words.length) queue.enqueue(i)
    val stack = Stack[Int]()
    var actions = ListBuffer[String]()
    var contexts = ListBuffer[Context]()
    var i = 0
    var action = "NA"
    while (queue.nonEmpty || stack.nonEmpty) {
      // case 1: stack forms a complete gold entity
      if (stack.nonEmpty) {
        val (u, v) = (stack.last, stack.top)
        if (spans.contains((u, v))) {
          val y = spans((u, v))
          action = s"RE-$y"
          contexts = contexts :+ Context(counter.incrementAndGet(), "bof", action, stack.clone(), queue.clone(), words, actions.toList)
          stack.clear()
        }
      } 
      if (action == "NA" && i < n) {
        // case 2: queue is not empty and token belongs to an entity
        if (spans.keySet.exists { case (l, r) => (l <= i) && (i <= r) }) {
          action = "SH"
          contexts = contexts :+ Context(counter.incrementAndGet(), "bof", action, stack.clone(), queue.clone(), words, actions.toList)
          queue.dequeue()
          stack.push(i)
        } else {
          action = "OUT"
          contexts = contexts :+ Context(counter.incrementAndGet(), "bof", action, stack.clone(), queue.clone(), words, actions.toList)
          queue.dequeue()
        }
        i = i + 1
      }
      actions = actions :+ action
      action = "NA"
    }
    contexts.toList
  }

  def main(args: Array[String]): Unit = {
    // test 1:
    // val words = Seq("Mark", "Watney", "visited", "Mars")
    // val spans = Map((0, 1) -> "PER", (3, 3) -> "LOC")
    // val contexts = decode(words, spans)
    // contexts.foreach(println)
    // (1,Stack(),Queue(0, 1, 2, 3)) => SH,List()
    // (2,Stack(0),Queue(1, 2, 3)) => SH,List(SH)
    // (3,Stack(0, 1),Queue(2, 3)) => RE-PER,List(SH, SH)
    // (4,Stack(),Queue(2, 3)) => OUT,List(SH, SH, RE-PER)
    // (5,Stack(),Queue(3)) => SH,List(SH, SH, RE-PER, OUT)
    // (6,Stack(3),Queue()) => RE-LOC,List(SH, SH, RE-PER, OUT, SH)    

    // test 2
    val words = Seq("John", "lives", "in", "New", "York")
    val spans = Map((0, 0) -> "PER", (3, 4) -> "LOC")
    val contexts = decode(Sample(words, spans))
    contexts.foreach(println)
    // (1,Stack(),Queue(0, 1, 2, 3, 4)) => SH,List()
    // (2,Stack(0),Queue(1, 2, 3, 4)) => RE-PER,List(SH)
    // (3,Stack(),Queue(1, 2, 3, 4)) => OUT,List(SH, RE-PER)
    // (4,Stack(),Queue(2, 3, 4)) => OUT,List(SH, RE-PER, OUT)
    // (5,Stack(),Queue(3, 4)) => SH,List(SH, RE-PER, OUT, OUT)
    // (6,Stack(3),Queue(4)) => SH,List(SH, RE-PER, OUT, OUT, SH)
    // (7,Stack(3, 4),Queue()) => RE-LOC,List(SH, RE-PER, OUT, OUT, SH, SH)    
  }


}

// repeat until (the stack and the buffer are both empty):
// The SHIFT transition moves a word from the buffer directly to the stack
// The OUT transition moves a word from the buffer directly into the output stack
// The REDUCE-y transition pops all items from the top of the stack creating a "chunk", labels with this label y 
//  and pushes a representation of this chunk onto the output stack.

// At each step:
//  If the stack matches a complete gold entity span → REDUCE
//  Else if the next buffer token is part of a gold entity → SHIFT
//  Else → OUT
// This priority ordering is crucial.

