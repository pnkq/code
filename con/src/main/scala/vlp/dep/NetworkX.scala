package vlp.dep
import collection.mutable.Set
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption

case class Node(id: String, attributes: Map[String, String])
case class Edge(src: String, tar: String, label: String)

object NetworkX {

  def createGraph(pathCoNLLU: String): (Set[Node], Set[Edge]) = {    
    val graphs = GraphReader.read(pathCoNLLU)
    val first = graphs.head
    // construct node set and edge set
    val nodePool = Set.empty[String] // containing unique node ids
    val nodes = Set.empty[Node]
    val edges = Set.empty[Edge]
    for (graph <- graphs) {
      val tokens = graph.sentence.tokens
      // first loop: add nodes
      for (token <- tokens) {
        // node
        val nodeId = token.word.toLowerCase + ":" + token.universalPartOfSpeech // "this:DET", "time:NOUN"
        val nodeAttributes = Map(
          "lemma" -> token.lemma, 
          "upos" -> token.universalPartOfSpeech, 
          "pos" -> token.partOfSpeech,
          "fs" -> token.featureStructure
        )
        if (!nodePool.contains(nodeId)) {
          nodes.add(Node(nodeId, nodeAttributes))
        }
      }
      // second loop: add edges
      for (token <- tokens.tail) {
        val head = tokens.find(_.id == token.head).get // cannot be None for a correct graph
        val source = head.word.toLowerCase + ":" + head.universalPartOfSpeech
        val target = token.word.toLowerCase + ":" + token.universalPartOfSpeech
        val label = token.dependencyLabel
        val edge = Edge(source, target, label)
        if (!edges.contains(edge)) {
          edges.add(edge)
        }
      }
    }
    println(s"There are ${nodes.size} nodes.")
    println(s"There are ${edges.size} edges.")
    (nodes, edges)
  }

  /**
   * Gets all targets (dependents) of a given source (head).
   * 
   * @param src
   * @param edges
   * @return a set of node ids
   */
  def targets(src: String, edges: Set[Edge]): Set[String] = {
    edges.filter(_.src == src).map(_.tar)
  }

  def main(args: Array[String]): Unit = {
    val path = "dat/dep/eng/UD_English-EWT/en_ewt-ud-train.conllu"
    val (nodes, edges) = createGraph(path)
    // export the edges to a TSV file
    val lines = edges.map { edge => edge.src + "\t" + edge.tar + "\t" + edge.label}.toList
    import scala.collection.JavaConverters._
    Files.write(Paths.get("dat/dep/eng-train.tsv"), lines.asJava, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }
}
