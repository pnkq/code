package vlp.dep


object GraphFilter {

  val words = Set("announced", "created", "troops", "analysis")

  def main(args: Array[String]): Unit = {
    val path = "dat/dep/UD_English-EWT/en_ewt-ud-dev.conllu"
    val graphs = GraphReader.read(path)

    def f(g: Graph): Boolean = {
      val n = g.sentence.tokens.size
      val ws = g.sentence.tokens.map(_.word)
      val b = ws.exists(w => words.contains(w))
      return (n <= 20) && (n >= 5) && b
    }
    val selection = graphs.filter(f)
    println("Number of graphs: " + selection.size)
    selection.take(20).foreach { graph => println(graph); println() }
  }
}
