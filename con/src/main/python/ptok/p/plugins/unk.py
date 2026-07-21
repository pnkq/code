from p.plugins.base import TokenizerPlugin
from p.piece import Piece

class UnknownPlugin(TokenizerPlugin):
    def __init__(self):
        super().__init__()

    def accepts(self, lang):
        return lang == "unk"

    def tokenize(self, token, return_pieces):
        if return_pieces:
            return [ Piece(text=token.text, source="unk", language="unk", start=token.start, end=token.end) ]
        else:
            return [ token[0] ]

