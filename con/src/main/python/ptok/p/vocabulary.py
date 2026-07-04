from collections import Counter
import json
from p.piece import PieceStat, PieceStatEncoder


class VocabularyBuilder:
    def __init__(self):
        self.counter = Counter()
        self.stats = {}

    def add_piece(self, piece):
        self.counter[piece.text] += 1
        self.stats[piece.text] = PieceStat(frequency=self.counter[piece.text], source=piece.source, language=piece.language)

    def add_stream(self, stream):
        for piece in stream:
            self.add_piece(piece)

    def build(self, min_frequency=2, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
        vocab = {}
        #
        # Special tokens first
        #
        for token in special_tokens:
            vocab[token] = len(vocab)

        #
        # Frequency order
        #
        for token, freq in self.counter.most_common():
            if freq < min_frequency:
                continue
            vocab[token] = len(vocab)

        return vocab

    def save(self, vocab, vocab_file, stat_file):
        with open(vocab_file, "w", encoding="utf8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        with open(stat_file, "w", encoding="utf8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2, cls=PieceStatEncoder)

