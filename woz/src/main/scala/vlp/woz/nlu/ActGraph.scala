package vlp.woz.nlu

import org.apache.spark.graphx.{EdgeRDD, VertexRDD}
import org.apache.spark.sql.{DataFrame, SparkSession}


/**
 * A tool to construct act graph.
 */
object ActGraph {

  private def createVertices(spark: SparkSession, df: DataFrame): VertexRDD[String] = {
    import spark.implicits._
    val vs = df.select("actNames").flatMap(row => row.getSeq[String](0)).distinct().collect()
    vs.foreach(println)
    println(vs.length)
    val is = (1L to vs.length)
    val pairRDD = spark.sparkContext.parallelize(is.zip(vs))
    VertexRDD(pairRDD)
  }

  private def createEdges(spark: SparkSession, df: DataFrame): EdgeRDD[Int] = {
    ???
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[4]").appName(ActGraph.getClass.getName).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val df = spark.read.json("dat/woz/nlu/dev")
    df.show()
    val vertices = createVertices(spark, df)
    println(vertices.toDebugString)

    spark.stop()
  }
}
