from p.piece import Piece
from transformers import AutoTokenizer


class EnglishTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize(self, span):
        # 1. Encode the text
        inputs = self.tokenizer(span.text, return_offsets_mapping=True)

        # 2. Extract the IDs and the Offsets from the dictionary
        input_ids = inputs["input_ids"]
        offsets = inputs["offset_mapping"]

        # 3. Convert the input IDs back into token strings
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        pieces = []
        for token, offset in zip(tokens, offsets):
            s, e = offset
            pieces.append(
                Piece(text=token, source="bpe", language="eng", start=span.start + s, end=span.start + e)
            )
        return pieces
    
