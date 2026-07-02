import re

from p.tokens import Token
from p.constants import Constants
from p.validator import VietnameseValidator


class Dispatcher:
    WORD = re.compile(r"\w+|[^\w\s]")
    PUNCT = re.compile(r"""[,.;:'?"`!@#$%^&*)()}{|~/\+=-]+""")
    NUMBER = re.compile(r"""\d+""")
    validator = VietnameseValidator("vietnamese_words.txt")

    def dispatch(self, text):
        result = []
        for m in self.WORD.finditer(text):
            word = m.group()
            if len(word) > 7:
                lang = "eng"
            elif re.search("[" + Constants.CONSONANTS_EN + "]", word.lower()):
                lang = "eng"
            elif re.search("[" + Constants.VOWELS_VI + "đ]", word.lower()):
                lang = "vie"
            elif self.validator.is_valid_syllable(word):
                lang = "vie"
            elif self.PUNCT.fullmatch(word):
                lang = "vie"
            elif self.NUMBER.fullmatch(word):
                lang = "vie"
            elif word.isascii():
                lang = "eng"
            else:
                lang = "unk"
            result.append(Token(word, lang, m.start(), m.end()))
        return result