package vlp.dep

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


/**
  * Merges graph features in separate files into one file.
  */
object Merger {
  def main(args: Array[String]): Unit = {
      // read graphX features
      val splits = Array("train", "dev", "test")
      val graphFeatures = Array("eigenvector-centrality", "indegree-centrality", "pagerank")
      val graphPaths = splits.map { split => graphFeatures.map(x => s"dat/dep/vie-$x-$split.tsv") }
      graphPaths.foreach(a => println(a.mkString("\t")))

      val spark = SparkSession.builder().master("local[*]").appName(Merger.getClass().getName()).getOrCreate()
      spark.sparkContext.setLogLevel("ERROR")

      val dfs = graphPaths.map { paths =>
        val af = paths.map { path => spark.read.option("delimiter", "\t").csv(path).toDF("t:p", "v") }
        val bf = af(0).union(af(1)).union(af(2))
        val gf = bf.groupBy("t:p").agg(collect_list("v"))
        gf.show(false)
        println("There are " + gf.count() + " tokens.")
      }
      spark.stop()
  }
}
