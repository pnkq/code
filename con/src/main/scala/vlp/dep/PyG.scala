package vlp.dep

import scala.io.Source
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption

/**
  * phuonglh
  * </p>
  * 
  * This utility reads in a graph in edge list (produced by NetworkX) and converts it 
  * to the PyG graph format, save it to a CSV file. The format of this CSV file is as follows:
    - the first line contains source vertex indices us of M edges (M integers)
    - the second line contains target vertex indices vs of M edges (M integers). (us[i], vs[i]) constitutes an arc of the graph.
    - the third line contains the label indices ys of N nodes. We use the uPOS as labels for each node.
    - next N lines contain features of N nodes in the graph. Each node has an array of features.
  */
object PyG {
  def main(args: Array[String]): Unit = {
    if (args.length == 0) {
      println("Need an argument of language: eng/fra/ind/vie")
    }
    val basePath = "dat/dep/"
    val language = args(0)
    val path = basePath + language + "-train.tsv"

    val source = Source.fromFile(path, "utf-8")
    val edges = source.getLines().toList.map { line =>
      val parts = line.split("""\s+""")
      (parts(0), parts(1))
    }
    source.close()

    println(s"Number of edges = ${edges.size}")


    val nodeSet = (edges.map { t => t._1 } ++ edges.map { t => t._2 }).distinct
    println(s"Number of nodes = ${nodeSet.size}")

    val labelSet = nodeSet.map { s =>
      val j = s.lastIndexOf(":")
      s.substring(j + 1)
    }.distinct
    println(s"Number of labels = ${labelSet.size}")

    edges.take(10).foreach(println)

    // convert nodes and edges to indices
    val node2Id = nodeSet.zipWithIndex.toMap
    val label2Id = labelSet.zipWithIndex.toMap
    val us = edges.map { t => node2Id(t._1) }
    val vs = edges.map { t => node2Id(t._2) }
    val ys = nodeSet.map { node =>
      val j = node.lastIndexOf(":")
      val label = node.substring(j + 1)
      (node2Id(node), label2Id(label))
    }.sortBy(_._1).map(_._2)

    val pathOut = basePath + language + "-pyg.tsv"
    Files.write(Paths.get(pathOut), us.mkString(" ").getBytes(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
    Files.write(Paths.get(pathOut), "\n".getBytes(), StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    Files.write(Paths.get(pathOut), vs.mkString(" ").getBytes(), StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    Files.write(Paths.get(pathOut), "\n".getBytes(), StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    Files.write(Paths.get(pathOut), ys.mkString(" ").getBytes(), StandardOpenOption.CREATE, StandardOpenOption.APPEND)

    val pathOutNode = basePath + language + "-nodeId.txt"
    val xs = nodeSet.mkString("\n")
    Files.write(Paths.get(pathOutNode), xs.getBytes(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }    
}
