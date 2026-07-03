from dataclasses import dataclass


@dataclass
class Piece:
    """
    One output token produced by a tokenizer.

    This is the basic unit that will become a vocabulary entry.
    """
    text: str
    # Which tokenizer produced this piece?
    source: str      # "vi", "bpe", "punct"
    # Original language of the span
    language: str    # "vi", "en", "punct"
    # Character offsets in the original text
    start: int
    end: int

    