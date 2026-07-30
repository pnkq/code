from p.plugins.base import TokenizerPlugin
from p.tokenizer_eng import EnglishTokenizer

class EnglishPlugin(TokenizerPlugin):
    def __init__(self):
        super().__init__()
        self.tokenizer = EnglishTokenizer()

    def accepts(self, lang):
        return lang == "eng"

    def tokenize(self, token, return_pieces):
        return self.tokenizer.tokenize(token, return_pieces)

