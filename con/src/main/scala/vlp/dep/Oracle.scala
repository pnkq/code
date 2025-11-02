package vlp.dep

import java.util.concurrent.atomic.AtomicInteger

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import org.apache.spark.ml.linalg.Vector


/**
  * Created by phuonglh on 6/22/17.
  * 
  * A parsing context contains a stack, a queue, an arc set and bag-of-features 
  *  (space-separated feature strings) and a transition label.
  * 
  */
case class Context(id: Int, bof: String, transition: String, 
  stack: mutable.Stack[String], queue: mutable.Queue[String], arcs: ListBuffer[Dependency], 
    words: Seq[String], tags: Seq[String], pastTransitions: Seq[String]) {
  override def toString: String = {
    val sb = new StringBuilder()
    sb.append('(')
    sb.append(id)
    sb.append(',')
    sb.append(bof)
    sb.append(',')
    sb.append(transition)
    sb.append(',')
    sb.append(stack)
    sb.append(',')
    sb.append(queue)
    sb.append(',')
    sb.append(arcs)
    sb.append(',')
    sb.append(words)
    sb.append(',')
    sb.append(tags)
    sb.append(',')
    sb.append(pastTransitions)
    sb.append(')')
    sb.toString()
  }
}

/**
  * Oracle is a key component in training transition-based parsers. It is used to derive 
  * optimal transition sequences from gold parse trees.
  *
  * @param featureExtractor
  */
abstract class Oracle(val featureExtractor: FeatureExtractor) {
  val counter = new AtomicInteger(0)

  def decode(graph: Graph): List[Context]

  /**
    * Derives all parsing contexts from a treebank of many dependency graphs.
    * @param graphs a list of manually-annotated dependency graphs.
    * @return a list of parsing contexts.
    */
  def decode(graphs: List[Graph]): List[Context] = {
    graphs.par.flatMap(graph => decode(graph)).toList
  }

}

/**
  * Created by phuonglh on 6/24/17.
  *
  * This is a static oracle for arc-eager transition-based dependency parsing. 
  * 
  * Decodes a manually-annotated dependency graph for 
  * parsing contexts and their corresponding transitions. This 
  * utility is used to create training data.
  *
  */
class OracleAE(featureExtractor: FeatureExtractor) extends Oracle(featureExtractor) {
  
  /**
    * Derives a transition sequence from this dependency graph. This 
    * is used to reconstruct the parsing process of a sentence.
    * 
    * @param graph a dependency graph
    * @return a list of parsing context
    */
  def decode(graph: Graph): List[Context] = {
    // create the initial config
    val stack = mutable.Stack[String]()
    val queue = mutable.Queue[String]()
    graph.sentence.tokens.foreach(token => queue.enqueue(token.id))
    stack.push(queue.dequeue())
    val arcs = ListBuffer[Dependency]()
    var config: Config = new ConfigAE(graph.sentence, stack, queue, arcs)
    val contexts = ListBuffer[Context]()
    while (!config.isFinal) {
      // extract features
      val features = featureExtractor.extract(config)
      // extract transition
      val u = stack.top
      val v = queue.front
      var transition = "SH"
      if (graph.hasArc(v, u)) {
        transition = "LA-" + graph.sentence.token(u).dependencyLabel
      } else if (graph.hasArc(u, v)) {
        transition = "RA-" + graph.sentence.token(v).dependencyLabel
      } else {
        // check precondition of the REDUCE transition
        if (config.isReducible(graph))
          transition = "RE"
      }
      val words = graph.sentence.tokens.map(_.word.toLowerCase()).toSeq
      val tags = graph.sentence.tokens.map(_.universalPartOfSpeech).toSeq
      val pastTransitions = contexts.map(_.transition).toSeq
      // add a parsing context      
      contexts += Context(counter.getAndIncrement(), features, transition, config.stack.clone(), 
        config.queue.clone(), config.arcs.clone(), words, tags, pastTransitions)
      config = config.next(transition)
    }
    contexts.toList
  }
}


/**
  * This is a static oracle for arc-standard transition-based dependency parsing.
  *
  * @param featureExtractor
  */
class OracleAS(featureExtractor: FeatureExtractor) extends Oracle(featureExtractor) {

  override def decode(graph: Graph): List[Context] = {
    // create the initial config
    val stack = mutable.Stack[String]()
    val queue = mutable.Queue[String]()
    graph.sentence.tokens.foreach(token => queue.enqueue(token.id))
    stack.push(queue.dequeue())
    val arcs = ListBuffer[Dependency]()
    var config: Config = new ConfigAS(graph.sentence, stack, queue, arcs)
    val contexts = ListBuffer[Context]()
    while (!config.isFinal) {
      // extract features
      val features = featureExtractor.extract(config)
      var transition = "SH" // default transition
      if (stack.nonEmpty && queue.nonEmpty) {
        // find the correct transition
        val (i, j) = (stack.top, queue.front)
        if (graph.hasArc(j, i)) {
          // precondition for LA: i is not ROOT and i does not have a head
          if ((i != "0") && !arcs.exists(a => a.dependent == i))
            transition = "LA-" + graph.sentence.token(i).dependencyLabel
        } else if (graph.hasArc(i, j)) {
          // precondition for RA: token j does not already have a head
          // RA will remove j but in the oracle mode, j is removed only if all remaining tokens do not take j as head
          if (!arcs.exists(a => a.dependent == j) && !queue.exists(k => graph.hasArc(j, k)))
            transition = "RA-" + graph.sentence.token(j).dependencyLabel
        }
      }
      val words = graph.sentence.tokens.map(_.word.toLowerCase()).toSeq
      val tags = graph.sentence.tokens.map(_.universalPartOfSpeech).toSeq
      val pastTransitions = contexts.map(_.transition).toSeq
      // add a parsing context
      contexts += Context(counter.getAndIncrement(), features, transition, config.stack.clone(), 
        config.queue.clone(), config.arcs.clone(), words, tags, pastTransitions)
      config = config.next(transition)
    }
    contexts.toList
  }
}