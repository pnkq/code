from p.plugins.base import TokenizerPlugin
from p.piece import Piece

class UnknownPlugin(TokenizerPlugin):
    def __init__(self):
        super().__init__()

    def accepts(self, span):
        return span.lang == "unk"

    def tokenize(self, span):
        return [ Piece(text=span.text, source="unk", language="unk", start=span.start, end=span.end) ]

