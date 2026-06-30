package vlp.sub

import scala.io.Source
import scala.collection.immutable.HashSet
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.charset.Charset
import java.nio.file.StandardOpenOption

/**
  * Filter mono-syllabic words from the Vietnamese lexicon which does not 
  * contain diacritics. This list of words are used to detect Vietnamese words 
  * from occidental (English, French) languages. 
  * 
  */
object SyllableFilter {
    val VIE = HashSet.empty[Char] ++ "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệòóỏõọìíỉĩịôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    val ENG = HashSet.empty[Char] ++ "fjwz"

    def main(args: Array[String]): Unit = {
        val lines = Source.fromFile("src/main/resources/lexicon.txt").getLines.filter {
            line => {
                val parts = line.split("\\s+").filter((_.nonEmpty))
                val hasAccent = parts(0).exists(c => VIE.contains(c))
                val hasEnglish = parts(0).exists(c => ENG.contains(c))
                (parts.length == 1) && (parts(0).length <= 7) && !hasAccent && !hasEnglish
            }
        }
        val output = lines.toList
        import scala.collection.JavaConverters._
        Files.write(Paths.get("src/main/resources/0.txt"), output.asJava, Charset.defaultCharset, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
    }
}
