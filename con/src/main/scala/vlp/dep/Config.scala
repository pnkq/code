package vlp.dep

import scala.collection.mutable
import scala.collection.mutable.ListBuffer


/**
 * A general config in a transition-based dependency parser. 
 * 
 */ 
abstract class Config(val sentence: Sentence, val stack: mutable.Stack[String], val queue: mutable.Queue[String], val arcs: ListBuffer[Dependency]) {
  def checkPrecondition(transition: String): Boolean
  def next(transition: String): Config
  def isReducible(): Boolean
  def isReducible(graph: Graph): Boolean
  def isFinal(): Boolean
  /**
    * Gets the raw sentence of this config.
    * @return a raw sentence.
    */
  def words: String = {
    "\"" + sentence.tokens.map(t => t.id + "/" + t.word + "/" + t.partOfSpeech + "/" + t.head).mkString(" ") + "\""
  }

  override def toString(): String = {
    val sb = new StringBuilder()
    sb.append('[')
    sb.append(words)
    sb.append(", ")
    sb.append(stack)
    sb.append(", ")
    sb.append(queue)
    sb.append(", ")
    sb.append(arcs)
    sb.append(']')
    sb.toString()
  }

}

/**
  * Created by phuonglh on 6/22/17.
  * 
  * A config in the arc-eager transition-based dependency parser.
  */
class ConfigAE(sentence: Sentence, stack: mutable.Stack[String], queue: mutable.Queue[String], arcs: ListBuffer[Dependency]) 
    extends Config(sentence, stack, queue, arcs) {

  def checkPrecondition(transition: String): Boolean = true // phuonglh: TODO
  /**
    * Computes the next config given a transition.
    * @param transition a transition
    * @return next config.
    */
  def next(transition: String): Config = {
    if (transition == "SH") {
      stack.push(queue.dequeue())
    } else if (transition == "RE") {
      stack.pop()
    } else if (transition.startsWith("LA")) {
      val u = stack.pop()
      val v = queue.front
      val label = transition.substring(3)
      arcs += Dependency(v, u, label)
    } else if (transition.startsWith("RA")) {
      val u = stack.top
      val v = queue.dequeue()
      stack.push(v)
      val label = transition.substring(3)
      arcs += Dependency(u, v, label)
    }
    new ConfigAE(sentence, stack, queue, arcs)
  }

  /**
    * Is this config reducible? The stack 
    * always contains at least the ROOT token, therefore, if the stack size is less than 1
    * then this config is not reducible. If the top element on the stack have not had a head yet, 
    * then this config is irreducible; otherwise, it is reducible.
    * @return true or false
    */
  def isReducible: Boolean = {
    if (stack.size < 1) false ; else {
      arcs.exists(d => d.dependent == stack.top)
    }
  }

  /**
    * A condition for RE transition in this static oracle algorithm. 
    * If config c = (s|i, j|q, A) and there exists k such that (1) k < i and (2) either (k -> j) 
    * or (j -> k) is an arc; then this config is reducible.
    * @param graph a gold graph
    * @return true or false.
    */
  def isReducible(graph: Graph): Boolean = {
    if (stack.size < 2) false; else {
      val i = stack.top
      val j = queue.front
      val allIds = graph.sentence.tokens.map(token => token.id)
      val iId = allIds.indexOf(i)
      var ok = false
      for (kId <- 0 until iId) {
        if (!ok) {
          val k = allIds(kId)
          if (graph.hasArc(k, j) || graph.hasArc(j, k))
            ok = true
        }
      }
      ok
    }
  }
  
  /**
    * Is this config final?
    * @return true or false
    */
  def isFinal: Boolean = queue.isEmpty || stack.isEmpty
}

/**
  * Configuration of an arc-standard transition-based parser. Implement the algorithm in Figure 3 of 
  * Joakim Nivre, "Algorithms for Deterministic Incremental Dependency Parsing", CL, 2008.
  *
  * @param sentence
  * @param stack
  * @param queue
  * @param arcs
  */
class ConfigAS(sentence: Sentence, stack: mutable.Stack[String], queue: mutable.Queue[String], arcs: ListBuffer[Dependency]) 
    extends Config(sentence, stack, queue, arcs) {

  def checkPrecondition(transition: String): Boolean = {
    if (transition == "SH") return queue.nonEmpty
    if (transition.startsWith("LA")) {
      if (stack.size <= 2) return false
      val secondTop = stack.toSeq(1)
      if (secondTop == "0") return false
      // secondTop does not already have a head (single-head constraint)
      return true // TODO
    }
    if (transition.startsWith("RA")) {
      if (stack.size <= 2) return false
      val top = stack.top
      // top does not already have a head (single-head constraint)
      return true // TODO
    }
    return true
  }
  override def next(transition: String): Config = {
    if (transition == "SH") {
      stack.push(queue.dequeue())
    } else {
      if (transition.startsWith("LA")) {
        val i = stack.pop()
        val j = queue.front
        val label = transition.substring(3)
        arcs += Dependency(j, i, label)
      } else if (transition.startsWith("RA")) {
        val i = stack.pop()
        val j = queue.dequeue()
        queue.+=:(i) // prepend the topmost element on the stack to the front of the queue
        val label = transition.substring(3)
        arcs += Dependency(i, j, label)
      }
    }
    return new ConfigAS(sentence, stack, queue, arcs)
  }

  override def isReducible(): Boolean = false

  override def isReducible(graph: Graph): Boolean = false

  override def isFinal(): Boolean = queue.isEmpty

}