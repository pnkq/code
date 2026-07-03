from p.constants import Constants
from p.piece import Piece

class VietnameseTokenizer:

    def tokenize(self, span: str) -> list[str]:
        """
        Breaks a Vietnamese syllable into sub-parts according to the Vietnamese syllable formation.
        """
        syllable = span.text
        s = syllable.lower()
        
        # Find the first vowel index
        i = 0        
        while i < len(s) and s[i] not in Constants.VOWELS_VI + Constants.VOWELS_EN:
            i += 1
            
        # Find the last vowel index
        j = len(s) - 1
        while j >= i and s[j] not in Constants.VOWELS_VI + Constants.VOWELS_EN:
            j -= 1
            
        # Slice the original syllable to preserve the original casing
        parts = [
            syllable[:i],        # Onset (initial consonant)
            syllable[i:j+1],     # Vowel nucleus
            syllable[j+1:]       # Coda (final consonant)
        ]
        
        # Filter out empty strings and return
        non_empty_parts = [p for p in parts if p]
        pieces = []
        cursor = span.start
        for w in non_empty_parts:
            pieces.append(
                Piece(text=w, source="vie", language="vie", start=cursor, end=cursor + len(w))
            )
            cursor += len(w)

        return pieces       

