package vlp.ner

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.charset.Charset
import java.nio.file.OpenOption
import java.nio.file.StandardOpenOption
import org.json4s.jackson.Serialization


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
    Files.write(Paths.get(pathOutput + "-ner"), lines.asJava, Charset.defaultCharset(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }

  def createSamples(path: String): Seq[Sample] = {
    ???
  }

  def main(args: Array[String]): Unit = {
    // create data for transition pretrainer
    val treebanks = Seq("eng.testa", "eng.testb")
    val dfs = treebanks.map { name =>
      val path = s"dat/ner/$name"
      println(name)
      val samples = createSamples(s"$path")
      run(samples, s"$path-ner.jsonl")
    }
  }
}
