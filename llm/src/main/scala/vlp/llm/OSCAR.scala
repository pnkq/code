package vlp.llm

import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import scopt.OptionParser

/**
 * OSCAR-2301 document extractor.
 * <p/>
 * This utility reads OSCAR file(s) into a data frame, tokenizes the content field, filters all documents
 * containing less than 80 syllables, flat map the contents into lines, filters line of length more than 40 characters
 * and less than 2048 characters; and writes the result into a text directory (20 partitions).
 *
 * <p/>
 * phuonglh
 *
 */

case class Config(
   master: String = "local[*]",
   totalCores: Int = 8, // X
   executorCores: Int = 4, // Y ==> there are X/Y executors
   executorMemory: String = "8g", // E
   driverMemory: String = "8g", // D
   level: String = "d", // d for document level, p for paragraph level
   version: String = "23",
   numPartitions: Int = 10,
   inputPath: String = "dat/23",
   outputPath: String = "pre/23",
   tempPath: String = "/tmp"
)

object OSCAR {

  final private val vietnameseCharacters: Set[Char] = Set(
    'à', 'á', 'ả', 'ã', 'ạ', 'â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ',
    'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ò', 'ó', 'ỏ', 'õ', 'ọ',
    'ì', 'í', 'ỉ', 'ĩ', 'ị', 'ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ',
    'ù', 'ú', 'ủ', 'ũ', 'ụ', 'ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'đ'
  )

  /**
   * Given a data frame, we remove all rows containing bad content (sex, code/script)
   * @param df a data frame
   * @param colName column of interest
   * @return a filtered data frame
   */
  def filterBadRows(df: DataFrame, colName: String): DataFrame = {
    df.filter(length(col(colName)) >= 80).filter(not(col(colName).contains("\u0000")))
  }

  /**
   * A line is considered not a proper Vietnamese text if there is not a Vietnamese-specific character
   * in a substring of a given length.
   * @param line a line to test
   * @param chunkLength length of chunk
   * @return true or false
   */
  private def isVietnamese(line: String, chunkLength: Int = 40): Boolean = {
    val chunks = line.toLowerCase.sliding(chunkLength)
    val containVietnamese = chunks.forall(p => p.toSet.exists(c => vietnameseCharacters.contains(c)))
    val containSpace = chunks.forall(p => p.contains(' '))
    containVietnamese && containSpace
  }

  /**
   * Preprocess the OSCAR 21 dataset containing plain text documents, where each document is separated by a newline. We need
   * to combine consecutive lines into a document. The documents are filtered to remove toxic contents and de-duplicated
   * at the document level.
   * @param spark spark session
   * @param cf a data frame
   * @return a data frame with a "text" column name.
   */
  private def f21(spark: SparkSession, cf: DataFrame): DataFrame = {
    val pairRDD = cf.rdd.zipWithIndex()
    val n = cf.count()
    val emptyLineIdx = pairRDD.map { case (row, id) =>
      val line = row.getAs[String](0).trim
      (line.length, id)
    }.filter(_._1 == 0).map(_._2.toInt).collect() ++ Array(n.toInt)
    // fill an index array for grouping consecutive lines into a document
    val ids = Array.fill[Int](n.toInt)(0)
    var start = 0
    for (i <- emptyLineIdx.indices) {
      val k = emptyLineIdx(i)
      (start until k).foreach(j => ids(j) = i)
      start = k
    }
    import spark.implicits._
    val pairDF = pairRDD.map(p => (p._1.getAs[String](0), p._2)).toDF("content", "lineIdx")
      .filter(length(trim(col("content"))) > 0)
    // zip the pairRDD with ids
    val idsRDD = spark.sparkContext.parallelize(ids).zipWithIndex()
    val idsDF = idsRDD.toDF("docIdx", "lineIdx")
    val df = pairDF.join(idsDF, "lineIdx")
      .withColumn("pair", concat_ws(";", col("lineIdx"), col("content")))
    val ef = df.groupBy("docIdx").agg(collect_list(col("pair")).alias("pairs"))
      .withColumn("xs", array_sort(col("pairs")))
    // sort a list of index-prepended sentences by their index
    // use parallel processing for speed-up
    val sorter = udf((xs: Seq[String]) => {
      xs.par.map { x =>
        val j = x.indexOf(";")
        (x.substring(0, j).toInt, x.substring(j+1))
      }.toList.sortBy(_._1).map(_._2)
    })
    val ff = ef.withColumn("ys", sorter(col("xs")))
      .withColumn("text", concat_ws("\n", col("ys")))
    filterBadRows(ff, "text").distinct()
  }

  /**
   * Preprocess the OSCAR 22/23 datasets containing JSONL documents. We filter bad rows and perform document deduplication.
   * @param spark spark session
   * @param cf a data frame which has a "content" raw text column.
   * @return a dataframe with a "text" column name.
   */
  private def f22(spark: SparkSession, cf: DataFrame): DataFrame = {
    print(s"Number of documents before filtering = ${cf.count}\n")
    val df = filterBadRows(cf, "content")
    // filter all documents having more than 80 tokens
    val tokenizer = new Tokenizer().setInputCol("content").setOutputCol("tokens")
    val ef = tokenizer.transform(df)
    val ff = ef.withColumn("size", size(col("tokens"))).filter(col("size") >= 80)
    print(s"Number of documents after filtering = ${ff.count}\n")
    // deduplicate the document (whole document level)
    val ffUnique = ff.select("content").distinct()
    // remove bad lines from each document before saving result
    println("Removing bad documents and then bad lines in each good document...")
    import spark.implicits._
    ffUnique.filter { row =>
      val content = row.getAs[String]("content")
      isVietnamese(content, 80)
    }.map { row =>
      row.getAs[String]("content").split("""\n+""")
        .filter(line => line.length >= 40 && line.length <= 2048)
        .filter(_.trim.nonEmpty)
        .filter(line => isVietnamese(line))
        .mkString("\n")
    }.toDF("text")
  }

  /**
   * Deduplicate documents at the paragraph level. Each document is split into a number of paragraphs.
   * Then these paragraphs are deduplicated. Unique paragraphs are merged back into document with the same
   * order. We need to label each paragraph with its document id and its own sequential id.
   * @param spark spark session
   * @param df a data frame
   * @param colName column of interest
   * @return a data frame with a "text" column name.
   */
  def distinctParagraph(spark: SparkSession, df: DataFrame, colName: String): DataFrame = {
    import spark.implicits._
    val pf = df.select(colName).rdd.zipWithIndex().flatMap { case (row, docId) =>
      val text = row.getAs[String](0)
      val lines = text.split("\n+").map(_.trim)
      val pairs = lines.zipWithIndex
      pairs.map { case (line, lineId) => ((docId.toInt, lineId), line) }
    }
    // create a data frame where each paragraph has an id = (docId, lineId)
    val ef = pf.toDF("id", "paragraph")
    println(s"Number of input paragraphs = ${ef.count}")
    // group the paragraphs by their content and collect their ids
    val gf = ef.groupBy("paragraph").agg(collect_list("id").as("ids"))
    // the ids columns contains an array of id, each id is a struct {docId, lineId}.
    // there are duplicates if the size of ids is greater than 1. We separate the head element out of the tail elements
    // note that the element starts from index 1, the "tailIds" column contains duplicated ids of the "headId" column
    val hf = gf.withColumn("id", element_at(col("ids"), 1))
      .withColumn("tailIds", slice(col("ids"), lit(2), size(col("ids"))))
      .select("id")
    // the hf data frame contains a single column of "id" that are unique. We then join it with the original ef data frame
    val jf = hf.join(ef, "id")
    println(s"Number of output paragraphs = ${jf.count}")
    // after deduplication, we split out the docId and lineId into separate columns
    val cf = jf.withColumn("docId", col("id._1")).withColumn("lineId", col("id._2"))
    // sort and group lines into list of paragraphs
    val bf = cf.sort("lineId").groupBy("docId").agg(collect_list("paragraph").as("lines"))
    // concat lines back into a newline-separated document
    bf.withColumn("text", concat_ws("\n", col("lines"))).select("text")
  }

  def main(args: Array[String]): Unit = {

    val opts = new OptionParser[Config](getClass.getName) {
      head(getClass.getName, "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory")
      opt[String]('E', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory")
      opt[String]('v', "version").action((x, conf) => conf.copy(version = x)).text("version 23/21/c4")
      opt[String]('i', "inputPath").action((x, conf) => conf.copy(inputPath = x)).text("input path (file/folder)")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path (file/folder)")
      opt[String]('t', "tempPath").action((x, conf) => conf.copy(tempPath = x)).text("temporary directory of Spark")
      opt[Int]('n', "numPartitions").action((x, conf) => conf.copy(numPartitions = x)).text("number of partitions for output")
      opt[String]('l', "level").action((x, conf) => conf.copy(level = x)).text("level to operate, either d (document) or p (paragraph)")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        val spark = SparkSession.builder().master(config.master)
          .config("spark.driver.memory", config.driverMemory)
          .config("spark.executor.memory", config.executorMemory)
          .config("spark.executor.cores", config.executorCores)
          .config("spark.deploy.defaultCores", config.totalCores)
          .config("spark.local.dir", config.tempPath)
          .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
          .appName("OSCAR").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        config.level match {
          case "d" =>
            config.version match {
              case "23" => // same for 22
                val cf = spark.read.option("recursiveFileLookup", value = true).json(config.inputPath).select("content")
                val gf = f22(spark, cf)
                print(s"Number of documents after filtering = ${gf.count}\n")
                gf.select("text").repartition(config.numPartitions).write.option("compression", "gzip").mode(SaveMode.Overwrite).json(config.outputPath)
              case "21" =>
                val cf = spark.read.option("recursiveFileLookup", value = true).text(config.inputPath).toDF("content")
                val df = f21(spark, cf)
                df.select("text").repartition(config.numPartitions).write.option("compression", "gzip").mode(SaveMode.Overwrite).json(config.outputPath)
                println(s"There are ${df.count()} documents.")
              case "c4" =>
                val df = spark.read.option("recursiveFileLookup", value = true).json(config.inputPath).select("text")
                val ef = filterBadRows(df, "text").distinct()
                ef.repartition(config.numPartitions).write.option("compression", "gzip").mode(SaveMode.Overwrite).json(config.outputPath)
                println(s"There are ${ef.count()} documents.")
              case "2x" =>
                // deduplication at the document level
                // combine and deduplicate all the 3 preprocessed sub-folders (21, 22, 23) of an inputPath)
                // this action needs a huge temporary disk memory, it is better to use the -t option to use a large NFS temporary folder
                val df = spark.read.option("recursiveFileLookup", value = true).json(config.inputPath)
                val ef = df.distinct().repartition(1)
                ef.write.mode(SaveMode.Overwrite).option("compression", "gzip").json(config.outputPath)
              case _ =>
                println("Require a version: [23, 21, 2x, c4]")
            }
          case "p" =>
            val colName = "text"
            val cf = spark.read.options(Map("recursiveFileLookup" -> "true")).json(config.inputPath).select(colName)
            println(s"Number of input documents = ${cf.count()}")
            val df = distinctParagraph(spark, cf, colName)
            println(s"Number of output documents = ${df.count()}")
            df.repartition(config.numPartitions).write.option("compression", "gzip").mode(SaveMode.Overwrite).json(config.outputPath)
        }
        spark.stop()
      case _ => println("Invalid options")
    }
  }
}
