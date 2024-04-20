package vlp.dsl

import java.io.{BufferedInputStream, FileInputStream}
import java.nio.file.{Files, Paths, StandardOpenOption}
import java.util.zip.GZIPInputStream
import scala.io.Source
import scala.collection.JavaConverters._

// phuonglh@gmail.com
object VocabMaker {

  /**
   * Reads a large embedding file (GloVe, Numberbatch, Fasttex, etc.) and filters
   * words using a vocab, writes result to a new embedding file.
   * @param vocab
   * @param embeddingPath
   * @param outputPath
   * @return
   */
  def vocabFilter(vocab: Set[String], embeddingPath: String, outputPath: String) = {
    val lines = if (!embeddingPath.endsWith(".gz")) {
      Source.fromFile(embeddingPath, "UTF-8").getLines()
    } else {
      val is = new GZIPInputStream(new BufferedInputStream(new FileInputStream(embeddingPath)))
      Source.fromInputStream(is).getLines()
    }
    val filtered = if (embeddingPath.contains("-EN-")) {
      lines.map { line =>
        val j = line.indexOf(" ")
        val word = line.substring(0, j)
        val rest = line.substring(j).trim
        (word, rest)
      }.filter(p => vocab.contains(p._1)).map(p => p._1 + " " + p._2).toList
    } else {
      // other languages, need to remove prefix, for example "/c/es/"
      lines.map { line =>
        val j = line.indexOf(" ")
        val word = line.substring(0, j).substring(6) // skip the common prefix
        val rest = line.substring(j).trim
        (word, rest)
      }.filter(p => vocab.contains(p._1)).map(p => p._1 + " " + p._2).toList
    }
    Files.write(Paths.get(outputPath), filtered.asJava, StandardOpenOption.CREATE_NEW, StandardOpenOption.TRUNCATE_EXISTING)
  }

  def main(args: Array[String]): Unit = {
    val language = args(0)
    if (Set("EN", "ES", "FR", "PT").contains(language)) {
      val vocab = Source.fromFile(s"dat/$language.vocab.txt").getLines().toSet
      val embeddingPath = s"dat/emb/numberbatch-$language-19.08.txt"
      vocabFilter(vocab, embeddingPath, s"dat/emb/numberbatch-$language-19.08.vocab.txt")
    } else {
      println("Unsupported language: " + args(0))
    }
  }
}
