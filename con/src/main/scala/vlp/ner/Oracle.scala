package vlp.ner

import scala.collection.mutable.{ListBuffer, Stack, Queue}
import java.util.concurrent.atomic.AtomicInteger


/**
  * Created by phuonglh on January 9, 2026.
  * 
  * A parsing context contains a stack, a queue, and bag-of-features 
  *  (space-separated feature strings) and a transition label.
  * 
  */
case class Context(id: Int, bof: String, transition: String, 
  stack: Stack[String], queue: Queue[String], 
    words: Seq[String], tags: Seq[String], pastTransitions: Seq[String]) {
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
    // sb.append(',')
    // sb.append(pastTransitions)
    sb.toString()
  }
}

/**
  * Oracle is a key component in training transition-based NER. It is used to derive 
  * optimal transition sequences from a gold sequence.
  *
  */
object Oracle {
  val counter = new AtomicInteger(0)

  def decode(words: Seq[String], labels: Seq[String]): List[Context] = {
    ???
  }
}
