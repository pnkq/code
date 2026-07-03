from tokenizers import Tokenizer
from p.piece import Piece
# from transformers import AutoTokenizer


class EnglishTokenizer:
    def __init__(self):
        # self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.tokenizer = Tokenizer.from_file("p/bpe_eng.json")

    # def tokenize(self, word):
    #     return self.tokenizer.tokenize(word)

    def tokenize(self, span):
        enc = self.tokenizer.encode(span.text)
        pieces = []

        for token, (s, e) in zip(enc.tokens, enc.offsets):
            pieces.append(
                Piece(text=token, source="bpe", language="eng", start=span.start + s, end=span.start + e)
            )
        return pieces
    
