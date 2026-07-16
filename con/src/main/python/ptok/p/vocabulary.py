from collections import Counter
import json
from json import JSONEncoder
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
                self.vocabulary.add(token=piece.text, source=piece.source, language=piece.language)
            
        return self.vocabulary        


@dataclass
class VocabularyEntry:
    id: int
    token: str
    source: str
    language: str
    frequency: int = 0


class Vocabulary:
    def __init__(self):
        # token -> VocabularyEntry
        self.token2entry = {}

        # id -> VocabularyEntry
        self.id2entry = {}

        # next available id
        self.next_id = 0 

    def add(self, token, source, language):
        if token in self.token2entry:
            self.token2entry[token].frequency += 1
            return self.token2entry[token]
        entry = VocabularyEntry(id=self.next_id, token=token, source=source, language=language, frequency=1)
        self.token2entry[token] = entry
        self.id2entry[self.next_id] = entry
        self.next_id += 1
        return entry
    
    def token_to_id(self, token):
        if token not in self.token2entry:
            return self.token2entry["<unk>"].id
        return self.token2entry[token].id

    def id_to_token(self, id):
        if id not in self.id2entry:
            return self.id2entry[self.token_to_id("<unk>")].token
        return self.id2entry[id].token

    
    def encode(self, pieces):
        ids = []
        for piece in pieces:
            ids.append(self.token_to_id(piece.text))
        return ids
    
    def decode(self, ids):
        pieces = []
        for idx in ids:
            entry = self.id_to_entry(idx)
            pieces.append(Piece(text=entry.token, source=entry.source, language=entry.language, start=-1, end=-1, word_id=-1))
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
            vocab.next_id = max(vocab.next_id, entry.id + 1)
        return vocab
    
    @classmethod
    def load_and_prune(cls, filename, min_frequency=5):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls()
        vocab.next_id = 0
        vocab.token2entry.clear()
        vocab.id2entry.clear()
        for item in data["entries"]:
            entry = VocabularyEntry(**item)
            if (entry.language == "pad") or (entry.frequency >= min_frequency):
              vocab.token2entry[entry.token] = entry
              vocab.id2entry[entry.id] = entry
              vocab.next_id = max(vocab.next_id, entry.id + 1)
        return vocab
    

class VocabularyEntryEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__
