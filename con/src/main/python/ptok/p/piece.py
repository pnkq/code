from dataclasses import dataclass
from json import JSONEncoder

@dataclass
class Piece:
    """
    One output token produced by a tokenizer.

    This is the basic unit that will become a vocabulary entry.
    """
    text: str
    # Which tokenizer produced this piece?
    source: str      # "vie", "bpe", "pad"
    # Original language of the span
    language: str    # "vie", "eng", "pad"
    # Character offsets in the original text
    start: int
    end: int
    # word id of this piece
    word_id: int = -1


@dataclass
class PieceStat:
    frequency: int = 0
    source: str = ""
    language: str = ""


class PieceStatEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__
    
