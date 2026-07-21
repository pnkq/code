from p.plugins.base import TokenizerPlugin
from p.tokenizer_vie import VietnameseTokenizer

class VietnamesePlugin(TokenizerPlugin):
    def __init__(self):
        super().__init__()
        self.tokenizer = VietnameseTokenizer()

    def accepts(self, lang):
        return lang == "vie"

    def tokenize(self, token, return_pieces):
        return self.tokenizer.tokenize(token, return_pieces)
    

