package vlp.irr

import org.apache.spark.sql.SparkSession

object GloVe {

  def main(args: Array[String]): Unit = {
    // Step 1: 
    val path = "/home/phuonglh/Downloads/glove.6B.50d.txt"
    val lines = scala.io.Source.fromFile(path).getLines().toList//.take(20000)
    // Step 2: create a GloVe map: {word -> array of doubles}
    val glove = lines.map { line => 
      val parts = line.split("\\s+")
      val word = parts.head
      val values = parts.tail.map(_.toDouble)
      (word, values)
    }.toMap

    println("Size of GloVe = " + glove.size)
    val bird = glove("paris")
    println(bird.mkString(", "))

    // Step 3
    def dot(u: Array[Double], v: Array[Double]) = {
      u.zip(v).map{ case (u_i, v_i) => u_i * v_i }.sum
    }

    def similarity(u: Array[Double], v: Array[Double]) = {
      dot(u, v) / (Math.sqrt(dot(u, u)) * Math.sqrt(dot(v, v)))
    }

    val ss = glove.par.map { case (k, v) => (k, similarity(bird, v)) }.toArray
    val xs = ss.sortBy(_._2).reverse
    val output = xs.take(20)
    output.foreach(println)
  }

}
