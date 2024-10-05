package vlp.dep

import scala.io.Source
import java.nio.file.Files
import java.nio.file.Paths
import _root_.java.nio.file.StandardOpenOption


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
      println("Need an argument of language: eng/fra/ind/vie.")
    }
    var basePath = "dat/dep/" 
    val language = args(0)
    val path = basePath + language + "-train.tsv"

    val edges = Source.fromFile(path, "utf-8").getLines().toList.map { line => 
      val parts = line.split("""\s+""")
      (parts(0), parts(1))
    }

    println(s"Number of edges = ${edges.size}")


    val nodeSet = (edges.map { t => t._1 } ++ edges.map {t => t._2 }).toSet.toList
    println(s"Number of nodes = ${nodeSet.size}")

    val labelSet = nodeSet.map { s => 
      val j = s.lastIndexOf(":")
      s.substring(j+1)
    }.toSet.toList
    println(s"Number of labels = ${labelSet.size}")

    edges.take(10).foreach(println)

    // convert nodes and edges to indices
    val node2Id = nodeSet.zipWithIndex.toMap
    val label2Id = labelSet.zipWithIndex.toMap
    val us = edges.map { t => node2Id(t._1) }
    val vs = edges.map { t => node2Id(t._2) }
    val ys = nodeSet.map { node =>
      val j = node.lastIndexOf(":")
      val label = node.substring(j+1)
      (node2Id(node), label2Id(label))
    }.toList.sortBy(_._1).map(_._2)

    val pathOut = basePath + language + "-pyg.tsv"
    Files.writeString(Paths.get(pathOut), us.mkString(" "), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
    Files.writeString(Paths.get(pathOut), "\n", StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    Files.writeString(Paths.get(pathOut), vs.mkString(" "), StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    Files.writeString(Paths.get(pathOut), "\n", StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    Files.writeString(Paths.get(pathOut), ys.mkString(" "), StandardOpenOption.CREATE, StandardOpenOption.APPEND)

    val pathOutNode = basePath + language + "-nodeId.txt"
    val xs = nodeSet.mkString("\n")
    Files.writeString(Paths.get(pathOutNode), xs, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }    
}
