from p.constants import Constants

class VietnameseTokenizer:

    def tokenize(self, syllable: str) -> list[str]:
        """
        Breaks a Vietnamese syllable into sub-parts according to the Vietnamese syllable formation.
        """
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
        return [p for p in parts if p]        
