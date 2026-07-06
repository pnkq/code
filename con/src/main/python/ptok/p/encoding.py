from dataclasses import dataclass, field
from typing import List, Tuple
from p.piece import Piece

@dataclass
class Encoding:
    # original normalized text
    text: str

    # Piece stream
    pieces: List[Piece] = field(default_factory=list)

    # Vocabulary ids
    ids: List[int] = field(default_factory=list)

    # Character offsets
    offsets: List[Tuple[int, int]] = field(default_factory=list)

    # Which lexical word produced this piece
    word_ids: List[int] = field(default_factory=list)

    # HuggingFace compatible
    attention_mask: List[int] = field(default_factory=list)

    special_tokens_mask: List[int] = field(default_factory=list)

    type_ids: List[int] = field(default_factory=list)  

    def validate(self):
        n = len(self.ids)
        assert len(self.pieces) == n
        assert len(self.offsets) == n
        assert len(self.word_ids) == n
        assert len(self.attention_mask) == n
        assert len(self.type_ids) == n
        assert len(self.special_tokens_mask) == n

        