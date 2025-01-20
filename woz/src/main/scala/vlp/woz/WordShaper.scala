package vlp.woz

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}

import scala.util.matching.Regex

/**
  * phuonglh@gmail.com
  *
  * Transform a sequence of words to a sequence of shapes. If a word does not have 
  * interested shapes, then it is NA: "abc" => "NA", "12" => "[NUMBER]", etc.
  * 
  */
class WordShaper(override val uid: String) extends UnaryTransformer[Seq[String], Seq[String], WordShaper] with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("shaper"))

  override protected def createTransformFunc: Seq[String] => Seq[String] = {
    _.map(x => {
      val sh = WordShape.shape(x)
      if (sh.isEmpty) "NA" else s"[${sh.toUpperCase}]"
    })
  }

  override protected def outputDataType: DataType = ArrayType(StringType, containsNull = false)
}

object WordShaper extends DefaultParamsReadable[WordShaper] {
  override def load(path: String): WordShaper = super.load(path)
}

/**
  * 
  * Detector of different word shapes.
  *
  */
object WordShape {
  private val allCaps: Regex = """\b\p{Lu}+([\s_]\p{Lu}+)*\b""".r
  private val number: Regex = """[+\-]?([0-9]*)?[0-9]+([.,]\d+)*""".r
  private val percentage: Regex = """[+\-]?([0-9]*)?[0-9]+([.,]\d+)*%""".r
  private val punctuation: Regex = """[.,?!;:"…/”“″'=^▪•<>&«\])(\[\u200b\ufeff+-]+""".r
  private val email: Regex = """(\w[-._%:\w]*@\w[-._\w]*\w\.\w{2,3})""".r
  private val url: Regex = """(((\w+)://)+[a-zA-z][\-\w]*\w+(\.\w[\-\w]*)+(/[\w\-~]+)*(\.\w+)?(/?)(\?(\w+=[\w%~]+))*(&(\w+=[\w%~]+))*|[a-z]+((\.)\w+)+)""".r
  private val date: Regex = """(\d\d\d\d)[-/.](\d?\d)[-/.](\d?\d)|((\d?\d)[-/.])?(\d?\d)[-/.](\d\d\d\d)""".r
  private val date1: Regex = """\b(([12][0-9]|3[01]|0*[1-9])[-/.](1[012]|0*[1-9])[-/.](\d{4}|\d{2})|(1[012]|0*[1-9])[-/.]([12][0-9]|3[01]|0*[1-9])[-/.](\d{4}|\d{2}))\b""".r
  private val date2: Regex = """\b(1[012]|0*[1-9])[-/](\d{4}|\d{2})\b""".r
  private val date3: Regex = """\b([12][0-9]|3[01]|0*[1-9])[-/](1[012]|0*[1-9])\b""".r
  private val time: Regex = """\b\d{1,2}:\d{1,2}\b""".r
  private val numberSeq: Regex = """\+?\d+([\s._-]+\d+){2,}\b""".r


  /**
    * Detects the shape of a word and returns its shape string or empty.
    * @param word a word
    * @return word shape
    */
  def shape(word: String): String = {
    word match {
      case email(_*) => "email"
      case url(_*) => "url"
      case allCaps(_*) => "allCaps"
      case date1(_*) => "date" // should goes after number (but we want to recognize 12.2004 as a date.)
      case date2(_*) => "date"
      case date3(_*) => "date"
      case date(_*) => "date"
      case time(_*) => "time"
      case numberSeq(_*) => "numSeq"
      case number(_*) => "number"
      case percentage(_*) => "percentage"
      case punctuation(_*) => "punctuation"
      case _ => ""
    }
  }
}