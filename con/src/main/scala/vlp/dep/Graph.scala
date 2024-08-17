package vlp.dep

/**
  * Created by phuonglh on 6/24/17.
  * 
  * A dependency graph for a sentence.
  */
case class Graph(sentence: Sentence) {
  
  /**
    * Gets the head -> dependent node id of the graph.
    * @return (token id -> its head id) map.
    */
  def heads: Map[String, String] = {
    var map = Map[String, String]()
    sentence.tokens.foreach(token => map += (token.id -> token.head))
    map
  }

  /**
    * Gets the (node -> dependency label) map of the graph 
    * @return (token id -> its dependency label) map.
    */
  def dependencies: Map[String, String] = {
    var map = Map[String, String]()
    sentence.tokens.foreach(token => map += (token.id -> token.dependencyLabel))
    map
  }

  /**
    * Checks whether this graph has an arc (u, v) or not. 
    * @param u a node id
    * @param v a node id
    * @return true or false
    */
  def hasArc(u: String, v: String): Boolean = {
    heads.exists(_ == (v, u))
  }

  /**
    * Checks whether there is an arc with a given dependency label going 
    * out from a token.
    * @param u token id
    * @param dependency dependency label 
    * @return true or false
    */
  def hasDependency(u: String, dependency: String): Boolean = {
    dependencies.exists(_ == (u, dependency))
  }

  def toText: String = {
    sentence.tokens.tail.map(_.lemma).mkString(" ")
  }

  override def toString: String = {
    val seq = dependencies.toSeq.sortBy(_._1.toInt).tail
    seq.map(pair => {
      val sb = new StringBuilder()
      val u = pair._1
      sb.append(pair._2)
      sb.append('(')
      sb.append(heads(u))
      sb.append('-')
      sb.append(sentence.token(heads(u)).word)
      sb.append(',')
      sb.append(u)
      sb.append('-')
      sb.append(sentence.token(u).word)
      sb.append(')')
      sb
    }).mkString("\n")
  }

  /**
   * Split a graph into 2 halves (leftGraph, rightGraph) using the root token.
   * @param graph a graph to split
   * @return two graphs (left, right), the ROOT token is shared between the two graphs
   */ 
  def splitGraph(graph: Graph): (Graph, Graph) = {
    val tokens = graph.sentence.tokens
    val rootToken = tokens.find(token => token.head == "0").get
    val rootIndex = tokens.indexOf(rootToken)
    // both left and right have the ROOT token
    val (leftTokens, rightTokens) = (tokens.take(rootIndex + 1), tokens.takeRight(tokens.size - rootIndex + 1))
    (Graph(Sentence(leftTokens)), Graph(Sentence(rightTokens)))
  }

}
