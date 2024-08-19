package vlp.dep

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


/**
  * Merges graph features in separate files into one file.
  * <p>
  * phuonglh@gmail.com, August 19, 2024.
  */
object Merger {
  def main(args: Array[String]): Unit = {
      // read graphX features
      val spark = SparkSession.builder().master("local[*]").appName(Merger.getClass().getName()).getOrCreate()
      spark.sparkContext.setLogLevel("ERROR")

      val splits = Array("train", "dev", "test")
      val graphFeatures = Array("eigenvector-centrality", "indegree-centrality", "pagerank")

      for (lang <- Array("eng", "ind", "vie")) {
        val graphPaths = splits.map { split => graphFeatures.map(x => s"dat/dep/$lang-$x-$split.tsv") }
        graphPaths.foreach(a => println(a.mkString("\t")))
        val dfs = graphPaths.map { paths =>
          val af = paths.map { path => spark.read.option("delimiter", "\t").option("inferSchema", true).csv(path).toDF("t:p", "v") }
          val bf = af(0).union(af(1)).union(af(2))
          bf.groupBy("t:p").agg(collect_list("v")).toDF("t:p", "graphX")
        }
        splits.zip(dfs).foreach { case (split, df) =>
          df.repartition(1).write.format("parquet").save(s"dat/dep/$lang-graphx-$split")
        }
      }

      spark.stop()
  }
}
