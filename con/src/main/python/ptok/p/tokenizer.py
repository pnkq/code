from p.dispatcher import Dispatcher

from p.tokenizer_vie import VietnameseTokenizer
from p.tokenizer_eng import EnglishTokenizer
from p.tokens import Token


class HybridTokenizer:

    def __init__(self):
        self.dispatcher = Dispatcher()
        self.vi = VietnameseTokenizer()
        self.en = EnglishTokenizer()

    def tokenize(self, text):
        pieces = []

        spans = self.dispatcher.dispatch(text)

        for span in spans:
            if span.lang == "vie":
                tokens = self.vi.tokenize(span.text)
                pieces.extend([Token(token, "vie", span.start, span.end) for token in tokens])
                # pieces.extend(self.vi.tokenize(span.text))
            elif span.lang == "eng":
                tokens = self.en.tokenize(span.text)
                pieces.extend([Token(token, "eng", span.start, span.end) for token in tokens])
                # pieces.extend(self.en.tokenize(span.text))
            else:
                pieces.append(span)
                # pieces.append(span.text)

        return pieces

    