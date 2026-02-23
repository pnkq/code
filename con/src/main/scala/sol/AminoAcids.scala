package sol

object AminoAcids {
    // The 20 Canonical Amino Acids
    val NON_POLAR = Map(
        "A" -> "Alanine",
        "V" -> "Valine",
        "L" -> "Leucine",
        "I" -> "Isoleucine",
        "M" -> "Methionine",
        "F" -> "Phenylalanine",
        "W" -> "Tryptophan",
        "P" -> "Proline",
        "G" -> "Glycine"
    )
    val CHARGED_POLAR = Map(
        "D" -> "Aspartate",
        "E" -> "Glutamate",
        "K" -> "Lysine",
        "R" -> "Arginine",
        "H" -> "Histidine",
    )
    val UNCHARGED_POLAR = Map(
        "S" -> "Serine",
        "T" -> "Threonine",
        "N" -> "Asparagine",
        "Q" -> "Glutamine",
        "Y" -> "Tyrosine",
        "C" -> "Cysteine",
    )
    val INDEX = Seq("A", "V", "L", "I", "M", "F", "W", "P", "G", "D", "E", "K", "R", "H", "S", "T", "N", "Q", "Y", "C")
        .zipWithIndex.map{case (c, i) => (c, i + 1)}.toMap
}
