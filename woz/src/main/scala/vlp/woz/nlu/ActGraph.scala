package vlp.woz.nlu

import org.apache.spark.graphx.{Edge, EdgeRDD, VertexId, VertexRDD}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, isnull, lag, not}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * A tool to construct act graph.
 *
 * phuonglh@gmail.com
 */
object ActGraph {

  /**
   * Each vertex is of type (id: Long, name: String)
   * @param spark Spark session
   * @param df a data frame containing act names
   * @return VertexRDD[String]
   */
  private def createVertices(spark: SparkSession, df: DataFrame): (RDD[(VertexId, String)], Map[String, Long]) = {
    import spark.implicits._
    val vs = df.select("actNames").flatMap(row => row.getSeq[String](0)).distinct().collect()
    val is = (1L to vs.length)
    val pairRDD = spark.sparkContext.parallelize(is.zip(vs))
    val map = vs.zip(is).toMap
    (VertexRDD(pairRDD), map)
  }

  /**
   * Each edge is of type (srcId: Long, tarId: Long, count: Int)
   * @param spark Spark session
   * @param df a data frame containing act names which are grouped by dialogId.
   * @param node2Id a map from node name to node id
   * @return EdgeRDD[Int]
   */
  private def createEdges(spark: SparkSession, df: DataFrame, node2Id: Map[String, Long]): RDD[Edge[Int]] = {
    // turn t: [a1, a2] => turn t+1: [a3, a4] ==> create 4 edges: [(a1, a3), (a1, a4), (a2, a3), (a2, a4)]
    val window = Window.partitionBy("dialogId").orderBy(col("turnId").cast("int"))
    // add 1 column for previous acts
    val df1 = df.withColumn("acts(-1)", lag("actNames", 1).over(window)).filter(not(isnull(col("acts(-1)"))))
    import spark.implicits._
    val acts = df1.select("acts(-1)", "actNames").flatMap { row =>
      val prevActs = row.getSeq[String](0)
      val currActs = row.getSeq[String](1)
      for (p <- prevActs; c <- currActs) yield Edge(node2Id(p), node2Id(c), 1)
    }.rdd
    // combine the counts on the same edge
    val combined = acts.groupBy(edge => (edge.srcId, edge.dstId))
      .mapValues(vs => vs.map(_.attr).sum).reduceByKey(_ + _)
      .map(triple => Edge(triple._1._1, triple._1._2, triple._2))
    EdgeRDD.fromEdges(combined)
  }

  private def test1(spark: SparkSession): Unit = {
    // take the first two dialogs and check the constructed act graph
    val dialogs = Seq(
      ("SNG1143.json", "0", Seq("ATTRACTION-INFORM")),
      ("SNG1143.json", "1", Seq("ATTRACTION-NOOFFER","ATTRACTION-REQUEST")),
      ("SNG1143.json", "2", Seq("ATTRACTION-INFORM")),
      ("SNG1143.json", "3", Seq("ATTRACTION-INFORM")),
      ("SNG1143.json", "4", Seq("ATTRACTION-REQUEST")),
      ("SNG1143.json", "5", Seq("ATTRACTION-INFORM","GENERAL-REQMORE")),
      ("SNG1143.json", "6", Seq("GENERAL-BYE")),
      ("SNG1143.json", "7", Seq("GENERAL-BYE")),
      ("SNG1142.json", "0", Seq("ATTRACTION-INFORM")),
      ("SNG1142.json", "1", Seq("ATTRACTION-INFORM","ATTRACTION-SELECT")),
      ("SNG1142.json", "2", Seq("ATTRACTION-REQUEST")),
      ("SNG1142.json", "3", Seq("ATTRACTION-INFORM","GENERAL-REQMORE")),
      ("SNG1142.json", "4", Seq("GENERAL-THANK")),
      ("SNG1142.json", "5", Seq("GENERAL-BYE"))
    )
    import spark.implicits._
    val df = spark.createDataset(dialogs).toDF("dialogId", "turnId", "actNames")
    val (vertices, map) = createVertices(spark, df)
    println(map)
    val edges = createEdges(spark, df, map)
    edges.foreach(println)
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[4]").appName(ActGraph.getClass.getName).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val df = spark.read.json("dat/woz/nlu/train")
    df.show()
    val (vertices, node2Id) = createVertices(spark, df)
    println(s"Number of nodes = ${node2Id.size}.")
    node2Id.foreach(println)
    val edges = createEdges(spark, df, node2Id)
    println(s"Number of edges = ${edges.count()}.")
//    edges.foreach(println)

    // print some stats of the act graph
    // max number of incoming/outgoing arcs to a node
    println("indegrees: ")
    val indegrees = edges.groupBy(_.srcId).map(pair => (pair._1, pair._2.size)).sortBy(_._2).collect()
    indegrees.foreach(println)

    println("outdegrees: ")
    val outdegrees = edges.groupBy(_.dstId).map(pair => (pair._1, pair._2.size)).sortBy(_._2).collect()
    outdegrees.foreach(println)

    // max/min count on each edge
    val (maxCount, minCount) = (edges.map(_.attr).max, edges.map(_.attr).min)
    println(s"maxCount = $maxCount, minCount = $minCount")

    val maxEdges = edges.sortBy(_.attr, ascending = false).take(5)
    maxEdges.foreach(println)

    spark.stop()
  }
}
