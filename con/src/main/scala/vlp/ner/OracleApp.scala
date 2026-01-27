package vlp.ner

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.charset.Charset
import java.nio.file.OpenOption
import java.nio.file.StandardOpenOption
import org.json4s.jackson.Serialization
import scala.io.Source
import scala.annotation.varargs


case class T(words: Seq[String], transitions: Seq[String])

/**
  * 
  * This utility constructs transition seqs from NE-tagged seqs.
  * 
  */
object OracleApp {
    
  def run(samples: Seq[Sample], pathOutput: String) = {
    import org.json4s._
    import org.json4s.jackson.Serialization
    val contexts = samples.map(sample => (sample, Oracle.decode(sample)))
    val lines = contexts.map { pair =>
      val sample = pair._1
      val words = sample.words
      val transitions = pair._2.last.pastTransitions
      Serialization.write(T(words, transitions))(org.json4s.DefaultFormats)
    }
    import scala.collection.JavaConverters._
    Files.write(Paths.get(pathOutput), lines.asJava, Charset.defaultCharset(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }

  def createSamples(path: String): Seq[Sample] = {
    ???
  }

  /**
    * Reads a NER corpus in CoNLL-2003 format.
    * @param dataPath
    * @return a list of sentences.
    */
  def readCoNLL(dataPath: String, twoColumns: Boolean = false): List[Sample] = {
    val lines = (Source.fromFile(dataPath, "UTF-8").getLines() ++ List("")).toArray
    val sentences = new ListBuffer[Sample]()
    val indices = lines.zipWithIndex.filter(p => p._1.trim.isEmpty).map(p => p._2)
    var u = 0
    var v = 0
    for (i <- (0 until indices.length)) {
      v = indices(i)
      if (v > u) { // don't treat two consecutive empty lines
        val ls = lines.slice(u, v)
        // extract words
        val words = ls.map(line => {
          val parts = line.trim.split("""\s+""")
          parts(0)
        })
        // extract spans
        val spans = mutable.Map[(Int, Int), String]()
        var j = u
        var s = 0
        var e = 0
        var prevLabel = "O"
        while (j < v) {
          val parts = lines(j).trim.split("""\s+""")
          val label = if (twoColumns) parts(1) else parts(3)
          if (label == "O") {
            if (s < e) { // there will be an entity in the range (s, e)
              spans.+=((s, e) -> prevLabel.substring(prevLabel.indexOf("-") + 1))
              s = e
            }
            s = s + 1
            e = e + 1
          } else {
            prevLabel = label
            e = e + 1
          }
          j = j + 1
        }
        sentences.append(Sample(words, spans.toMap))
      }
      u = v + 1
    }
    sentences.toList
  }

  def main(args: Array[String]): Unit = {
    // create data for transition pretrainer
    val treebanks = Seq("syll.txt")
    val dfs = treebanks.map { name =>
      val samples = readCoNLL(s"dat/med/$name")
      // samples.foreach(println)
      run(samples, s"dat/med/$name.jsonl")
    }
  }
}
