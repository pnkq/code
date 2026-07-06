from abc import ABC, abstractmethod
from p.piece import Piece
from p.encoding import Encoding

class PostProcessor(ABC):
    @abstractmethod
    def process(self, encoding: Encoding) -> Encoding:
        pass

from encoding import Encoding
from postprocessor import PostProcessor


class RoBERTaPostProcessor(PostProcessor):

    def __init__(self, vocabulary):
        self.vocab = vocabulary
        self.cls_id = vocabulary.token_to_id("<s>")
        self.sep_id = vocabulary.token_to_id("</s>")

    def _wrap(self, values, left, right):
        return [left] + values + [right]        

    def process(self, encoding: Encoding):
        # ids
        encoding.ids = self._wrap(encoding.ids, self.cls_id, self.sep_id)
        
        # attention mask
        encoding.attention_mask = self._wrap(encoding.attention_mask, 1, 1)

        # special token mask
        encoding.special_tokens_mask = self._wrap(encoding.special_tokens_mask, 1, 1)

        # type ids
        encoding.type_ids = self._wrap(encoding.type_ids, 0, 0)

        # offsets
        encoding.offsets = self._wrap(encoding.offsets, (-1,-1), (-1,-1))

        # word ids
        encoding.word_ids = self._wrap(encoding.word_ids, -1, -1)

        # updated pieces
        cls_piece = Piece(text="<s>", source="special", language="special", start=-1, end=-1, word_id=-1)
        sep_piece = Piece(text="</s>", source="special", language="special", start=-1, end=-1, word_id=-1)
        
        encoding.pieces = self._wrap(encoding.pieces, cls_piece, sep_piece)

        return encoding
    

# class PostProcessor:
#     def process(self, pieces):
#         output = []
#         output.append(Piece(text="<s>", source="special", language="special", start=-1, end=-1))
#         output.extend(pieces)
#         output.append(Piece(text="</s>", source="special", language="special", start=-1, end=-1))
#         return output
    
