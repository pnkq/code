from itertools import groupby
from p.encoding import Encoding


SPECIAL = {
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "<mask>"
}


class Decoder:
    def __init__(self, vocabulary):
        self.vocab = vocabulary

    def decode(self, encoding):
        # Remove special tokens
        pieces = [ p for p in encoding.pieces if p.text not in SPECIAL ]

        # This reconstructs words, not typography
        words = []
        for _, group in groupby(pieces, key=lambda p: p.word_id):
            word = "".join(piece.text for piece in group)
            words.append(word)
        return " ".join(words)
    
    def reconstruct(self, encoding):
        return encoding.text
    
    def decode_ids(self, ids):
        tokens = []
        for idx in ids:
            token = self.vocab.id_to_token(idx)
            if token in SPECIAL:
                continue
            tokens.append(token)
        return self.decode_tokens(tokens)
    
    def decode_tokens(self, tokens):
        return " ".join(tokens)
    
    

# class Decoder:
    # def decode(self, pieces):
    #     words = []
    #     pieces = [p for p in pieces if p.language != 'special']
    #     # decode by using offsets
    #     pieces.sort(key=lambda p: p.start)
    #     text = ""
    #     cursor = 0

    #     for piece in pieces:
    #         while cursor < piece.start:
    #             text += " "
    #             cursor += 1
    #         text += piece.text
    #         cursor = piece.end
    #     return text

