from p.dispatcher import Dispatcher

from p.tokenizer_vie import VietnameseTokenizer
from p.tokenizer_eng import EnglishTokenizer
from p.tokens import Token
from p.piece import Piece


class HybridTokenizer:

    def __init__(self):
        self.dispatcher = Dispatcher()
        self.vie = VietnameseTokenizer()
        self.eng = EnglishTokenizer()

    def tokenize(self, text):
        pieces = []

        spans = self.dispatcher.dispatch(text)

        for span in spans:
            if span.lang == "vie":
                tokens = self.vie.tokenize(span)
                pieces.extend(tokens)
            elif span.lang == "eng":
                tokens = self.eng.tokenize(span)
                pieces.extend(tokens)
            else:
                pieces.append(Piece(text=span.text, source="unk", language="unk", start=span.start, end=span.end))

        return pieces

    