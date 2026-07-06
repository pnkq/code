from encoding import Encoding

class Encoder:

    def __init__(self, vocabulary, postprocessor):
        self.vocab = vocabulary
        self.postprocessor = postprocessor

    def encode(self, text, pieces):
        encoding = Encoding(text=text)
        encoding.pieces = pieces

        # Vocabulary lookup
        encoding.ids = [ self.vocab.token_to_id(piece.text) for piece in pieces ]

        # Offsets
        encoding.offsets = [ (piece.start, piece.end) for piece in pieces ]

        # Word ids
        encoding.word_ids = [ piece.word_id for piece in pieces ]

        # Attention mask
        encoding.attention_mask = [ 1 for _ in pieces ]

        # Type ids
        encoding.type_ids = [ 0 for _ in pieces ] 

        # Special tokens
        encoding.special_tokens_mask = [ 0 for _ in pieces ] 

        # Insert <s>, </s>
        encoding = self.postprocessor.process(encoding)

        return encoding