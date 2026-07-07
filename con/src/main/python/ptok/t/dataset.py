from t.corpus import CorpusReader
from t.encoder import CorpusEncoder
from t.packer import SequencePacker
import numpy as np


class DatasetBuilder:
    def __init__(self, pipeline, vocab, corpus_dir, max_length=512):
        self.reader = CorpusReader(corpus_dir)
        self.encoder = CorpusEncoder(pipeline, vocab)
        self.packer = SequencePacker(max_length)

    def build(self):
        for doc in self.reader.documents():
            ids = self.encoder.encode_document(doc)
            yield from self.packer.add(ids)
        yield from self.packer.flush()

    def save(self, filename):
        sequences = list(self.build())
        print(sequences[0])
        print(sequences[-1])
        np.save(filename, np.array(sequences, dtype=np.int32))

