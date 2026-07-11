from p.plugins.base import TokenizerPlugin
from p.tokenizer_vie import VietnameseTokenizer

class VietnamesePlugin(TokenizerPlugin):
    def __init__(self):
        super().__init__()
        self.tokenizer = VietnameseTokenizer()

    def accepts(self, span):
        return span.lang == "vie"

    def tokenize(self, span):
        return self.tokenizer.tokenize(span)

