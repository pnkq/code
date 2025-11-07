package vlp.dep

import org.json4s._
import org.json4s.jackson.Serialization    
import scala.collection.JavaConverters._
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.charset.Charset
import java.nio.file.StandardOpenOption


case class S(words: Seq[String], tags: Seq[String], heads: Seq[String], labels: Seq[String])

/**
  * phuonglh@gmail.com
  * 
  */

object GraphExporter {
  def main(args: Array[String]): Unit = {
    val splits = Seq("train", "dev", "test")
    splits.foreach { split =>
      val pathCoNLLU = s"dat/dep/UD_English-EWT/en_ewt-ud-$split.conllu"
      val graphs = GraphReader.read(pathCoNLLU)
      val lines = graphs.map { graph =>
        val words = graph.sentence.tokens.tail.map(_.word)
        val tags = graph.sentence.tokens.tail.map(_.universalPartOfSpeech)
        val heads = graph.sentence.tokens.tail.map(_.head)
        val labels = graph.sentence.tokens.tail.map(_.dependencyLabel)
        Serialization.write(S(words, tags, heads, labels))(org.json4s.DefaultFormats)
      }
      val pathOutput = s"dat/dep/UD_English-EWT/en_ewt-ud-$split.jsonl"
      Files.write(Paths.get(pathOutput), lines.asJava, Charset.defaultCharset(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
    }
  }
}
