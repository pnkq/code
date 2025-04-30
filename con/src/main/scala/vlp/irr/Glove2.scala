package vlp.irr

object Glove2 {
  def main(args: Array[String]): Unit = {
    // Step 1: Read all word vectors
    val path = "/home/phuonglh/Downloads/glove.6B.50d.txt"
    val lines = scala.io.Source.fromFile(path).getLines().toList
    // Step 2: Convert to map {key -> vector}
    val glove = lines.par.map { line => 
      val parts = line.split("\\s+")
      val word = parts.head
      val vector = parts.tail.map(_.toDouble)
      (word, vector)
    }.toMap
    println("Number of keys = " + glove.size)
    val bird = glove("bird")
    println(bird.mkString(", "))

    // Step 3: 
    def dot(u: Array[Double], v: Array[Double]) = {
      u.zip(v).map { case (u_i, v_i) => u_i * v_i }.sum
    }

    def similarity(u: Array[Double], v: Array[Double]) = {
      dot(u, v) / (Math.sqrt(dot(u, u)) * Math.sqrt(dot(v, v)))
    }

    val ss = glove.par.map { case (k, v) => (k, similarity(bird, v)) }.toArray
    val output = ss.sortBy(_._2).reverse.take(20)
    output.foreach(println)

  }
}
