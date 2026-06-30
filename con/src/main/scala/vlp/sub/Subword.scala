package vlp.sub

import scala.collection.immutable.HashSet

/**
 * Vietnamese-specific subword tokenizer. 
 * 
 * June 29, 2026
 * 
 * phuonglh@gmail.com 
 * 
 * */

object Subword {
    val vowels = HashSet.empty[Char] ++ "aàáảãạâầấẩẫậăằắẳẵặeèéẻẽẹêềếểễệoòóỏõọiìíỉĩịôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ"

    /**
     * Breaks a Vietnamese syllable into sub-parts according to the Vietnamese syllable formation.
     */
    def split(syllable: String): Seq[String] = {
        // all Vietnamese syllables have less than 8 characters
        if (syllable.length > 7) return Array(syllable)

        // find the first vowel index
        val s = syllable.toLowerCase()
        var i = 0        
        while (i < s.length && !vowels.contains(s.charAt(i))) i = i + 1
        // find the last vowel index
        var j = s.length - 1
        while (j >= i && !vowels.contains(s.charAt(j))) j = j - 1
        val parts = Array(syllable.substring(0, i), syllable.substring(i, j+1), syllable.substring(j+1, syllable.length))
        return parts.filter(_.nonEmpty)
    }

    def main(args: Array[String]): Unit = {
        val syllables = ("92 năm cuộc đời tận hiến cho ngành y, BS Nguyễn Đình Hối khai sinh mô hình " +
          "bệnh viện đại học đầu tiên tại Việt Nam, gieo vào lòng nhiều thế hệ bác sĩ triết lý học để biết dừng dao mổ. " + 
          "Sinh năm 1936 tại Thanh Hóa, trong một gia đình khoa bảng, Nguyễn Đình Hối từng đạp xe hơn 200 km ra Hà Nội dự thi" +
          " đại học và đỗ thủ khoa cả hai trường Sư phạm và Y khoa năm 1954. Chọn nghề y theo tâm nguyện của gia đình, cái duyên" + 
          " sư phạm vẫn gắn chặt với ông suốt cuộc đời. Âm tiết dài nhất có 7 ký tự là từ nghiêng. The well-beings of what?"
          )
          .split("""[,.\s]+""")
        syllables.foreach(s => {
            println(split(s))
        })
        println("===")
        syllables.foreach(s => {
            println(split(s.toUpperCase()))
        })
    }
}
