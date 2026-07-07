from collections import Counter
import json
from dataclasses import dataclass, asdict
from p.piece import Piece


class VocabularyBuilder:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.vocabulary = Vocabulary()
    def initialize(self):
        self.vocabulary.add("<pad>", "pad", "pad")
        self.vocabulary.add("<unk>", "pad", "pad")
        self.vocabulary.add("<s>", "pad", "pad")
        self.vocabulary.add("</s>", "pad", "pad")
        self.vocabulary.add("<mask>", "pad", "pad")

    def build(self, corpus_reader):
        self.initialize()
        for document in corpus_reader.documents():
            pieces = self.pipeline.tokenize(document)
            for piece in pieces:
                self.vocabulary.add(piece.text, piece.source, piece.language)
        return self.vocabulary        


@dataclass
class VocabularyEntry:
    id: int
    token: str
    source: str
    language: str
    can_start_word: bool = True


class Vocabulary:
    def __init__(self):
        # token -> VocabularyEntry
        self.token2entry = {}

        # id -> VocabularyEntry
        self.id2entry = {}

        # next available id
        self.next_id = 0 

    def add(self, token, source, language, can_start_word=True):
        if token in self.token2entry:
            return self.token2entry[token]

        entry = VocabularyEntry(id=self.next_id, token=token, source=source, language=language, can_start_word=can_start_word)
        self.token2entry[token] = entry
        self.id2entry[self.next_id] = entry
        self.next_id += 1
        return entry
    
    def token_to_id(self, token):
        if token not in self.token2entry:
            return self.token2entry["<unk>"].id
        return self.token2entry[token].id        
    
    def encode(self, pieces):
        ids = []
        for piece in pieces:
            ids.append(self.token_to_id(piece.text))
        return ids
    
    def decode(self, ids):
        pieces = []
        for idx in ids:
            entry = self.id_to_entry(idx)
            pieces.append(
                Piece(text=entry.token, source=entry.source, language=entry.language, start=-1, end=-1, word_id=-1))
        return pieces
    
    def __contains__(self, token):
        return token in self.token2entry
    
    def __len__(self):
        return len(self.id2entry)
    
    @property
    def vocab_size(self):
        return len(self)
    
    def save(self, filename):
        data = {
            "version": "1.0",
            "vocab_size": len(self),
            "entries": [ asdict(entry) for entry in self.id2entry.values() ]
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls()
        vocab.next_id = 0
        vocab.token2entry.clear()
        vocab.id2entry.clear()
        for item in data["entries"]:
            entry = VocabularyEntry(**item)
            vocab.token2entry[entry.token] = entry
            vocab.id2entry[entry.id] = entry
            vocab.next_id = max(
                vocab.next_id,
                entry.id + 1
            )
        return vocab    
    
    