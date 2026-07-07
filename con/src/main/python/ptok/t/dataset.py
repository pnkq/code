from t.corpus import CorpusReader
from t.encoder import CorpusEncoder
from t.packer import SequencePacker
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

class DatasetBuilder:
    def __init__(self, pipeline, vocab, corpus_dir, max_length=512):
        self.reader = CorpusReader(corpus_dir)
        self.encoder = CorpusEncoder(pipeline, vocab)
        self.packer = SequencePacker(max_length)
        self.stats = BuildStats()

    def build(self):
        progress = tqdm(unit=" pieces")
        for doc in self.reader.documents():
            self.stats.documents += 1
            ids = self.encoder.encode_document(doc)
            self.stats.pieces += len(ids)
            progress.update(len(ids))
            yield from self.packer.add(ids)
        self.packer.flush()
        progress.close()

    def save(self, filename):
        sequences = list(self.build())
        np.save(filename, np.array(sequences, dtype=np.int32))



@dataclass
class BuildStats:
    documents: int = 0
    pieces: int = 0
