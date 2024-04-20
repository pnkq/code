package vlp.dsl

import java.nio.file.{Files, Paths, StandardOpenOption}
import scala.io.Source

/**
 * Preprocess the French corpus so as to have the same format as the other corpora.
 */
object French {
  val basePath = "dat/DSL-ML-2024/FR/"

  private def convert(split: String): Unit = {
    var source = Source.fromFile(basePath + s"$split.labels")
    val devLabels = source.getLines().toList
    source.close()
    source = Source.fromFile(basePath + s"$split.txt")
    val devTexts = source.getLines().toList
    source.close()
    val lines = devLabels.zip(devTexts).map { pair =>
      pair._1.trim + "\t" + pair._2.trim
    }
    val outputPath = basePath + s"FR_$split.tsv"
    import scala.jdk.CollectionConverters._
    Files.write(Paths.get(outputPath), lines.asJava, StandardOpenOption.CREATE_NEW, StandardOpenOption.TRUNCATE_EXISTING)

  }
  def main(args: Array[String]): Unit = {
    convert("dev")
    convert("train")
  }
}
