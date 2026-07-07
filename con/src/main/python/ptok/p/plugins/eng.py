from p.plugins.base import TokenizerPlugin
from p.tokenizer_eng import EnglishTokenizer

class EnglishPlugin(TokenizerPlugin):
    def __init__(self):
        super().__init__()
        self.tokenizer = EnglishTokenizer()

    def accepts(self, span):
        return span.lang == "eng"

    def tokenize(self, span):
        return self.tokenizer.tokenize(span)

