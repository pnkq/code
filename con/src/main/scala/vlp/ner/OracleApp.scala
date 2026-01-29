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
      val context = pair._2.last
      val transitions = context.pastTransitions :+ context.transition
      Serialization.write(T(words, transitions))(org.json4s.DefaultFormats)
    }
    import scala.collection.JavaConverters._
    Files.write(Paths.get(pathOutput), lines.asJava, Charset.defaultCharset(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }

  def createSamples(path: String): Seq[Sample] = {
    ???
  }

  def extractSpans(labels: Seq[String], offset: Int): Map[(Int, Int), String] = {
    if (labels.length > 0) {
      var s = 0
      // find the first start index of an entity
      while (s < labels.size && !labels(s).startsWith("B-")) s = s + 1
      // find the end index of this entity
      var e = s + 1
      while (e < labels.size && labels(e).startsWith("I-")) e = e + 1
      // (s, e) is the boundary of the entity
      if (s < labels.size) {
        val j = labels(s).indexOf("-")
        val entityName = labels(s).substring(j + 1)
        return extractSpans(labels.slice(e, labels.length), offset + e) + ((s + offset, e - 1 + offset) -> entityName)
      }
      else return Map[(Int, Int), String]()
    } else {
      return Map[(Int, Int), String]()
    }
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
        val pairs = ls.map(line => {
          val parts = line.trim.split("""\s+""")
          (parts(0), if (twoColumns) parts(1) else parts(3))
        })
        val words = pairs.map(_._1)
        val labels = pairs.map(_._2)
        val spans = extractSpans(labels, 0)
        sentences.append(Sample(words, spans))
      }
      u = v + 1
    }
    return sentences.toList
  }

  def main(args: Array[String]): Unit = {
    // create data for transition pretrainer
    val treebanks = Seq("covid19", "vietmedner", "vimedner", "vimq")
    val dfs = treebanks.map { name =>
      val samples = readCoNLL(s"dat/med/$name.conll", true)
      run(samples, s"dat/med/$name.jsonl")
    }
  }
}
