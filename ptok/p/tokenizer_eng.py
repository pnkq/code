from p.piece import Piece
from transformers import AutoTokenizer


class EnglishTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize(self, token, return_pieces):
        # 1. Encode the text
        text = ""
        if return_pieces:
            # the token is of type Token(text, lang, start, end)
            text = token.text
        else:
            # the token is a pair (text, lang)
            text = token[0]

        inputs = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

        # 2. Extract the IDs and the Offsets from the dictionary
        input_ids = inputs["input_ids"]
        offsets = inputs["offset_mapping"]

        # 3. Convert the input IDs back into token strings
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        if return_pieces:
            for tok, offset in zip(tokens, offsets):
                s, e = offset
                yield Piece(text=tok, source="bpe", language="eng", start=token.start + s, end=token.start + e)
        else:
            for token in tokens:
                yield token
        
    
