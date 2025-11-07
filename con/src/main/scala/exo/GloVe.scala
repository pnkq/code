package exo

object GloVe {
  def main(args: Array[String]): Unit = {
    val path = "/Users/phuonglh/glove.6B.50d.txt"
    val lines = scala.io.Source.fromFile(path).getLines().toList
    val vocab = lines.map { line =>
      val parts = line.split("\\s+")
      val word = parts.head
      val vector = parts.tail.map(_.toDouble)
      (word, vector)
    }.toMap
    println("Size of vocab = " + vocab.size)
    val v = vocab("bird")
    println(v.mkString(","))

    def dot(u: Array[Double], v: Array[Double]) = {
      u.zip(v).map(pair => pair._1 * pair._2).sum
    }
    def cosine(u: Array[Double], v: Array[Double]) = {
      dot(u, v) / (Math.sqrt(dot(u, u)) * Math.sqrt(dot(v, v)))
    }
    val ss = vocab.par.map { case (word, vector) => (word, cosine(vector, v)) }.toArray
    val output = ss.sortBy(_._2).reverse.take(20)
    output.foreach(println)
  }
}
