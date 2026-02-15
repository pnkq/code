package sol

object AminoAcids {
    // The 20 Canonical Amino Acids
    val NON_POLAR = Map(
        "A" -> "Alanine",
        "V" -> "Valine",
        "L" -> "Leucine",
        "I" -> "Isoleucine",
        "M" -> "Methionine",
        "P" -> "Phenylalanine",
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
}
