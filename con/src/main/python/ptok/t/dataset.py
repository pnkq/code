from t.corpus import CorpusReader
from t.packer import SequencePacker
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass


# Version 2. Parallel processing, faster.
class DatasetBuilderP:

    def __init__(self, tokenizer, corpus_dir, sequence_length=512, drop_last=True):
        self.tokenizer = tokenizer
        self.reader = CorpusReader(corpus_dir)
        self.packer = SequencePacker(sequence_length)
        self.drop_last = drop_last
        self.stats = BuildStats()

    def build(self):
        progress = tqdm(unit=" tokens")

        for doc in self.reader.documents():
            ids = self.tokenizer.encode(doc)
            self.stats.lines += 1
            self.stats.pieces += len(ids)

            progress.update(len(ids))

            for seq in self.packer.add(ids):
                yield self._finalize(seq)

        for seq in self.packer.flush(drop_last=self.drop_last, pad_id=self.tokenizer.pad_token_id):
            yield self._finalize(seq)

        progress.close()

    def _finalize(self, seq):
        self.stats.sequences += 1
        return self.tokenizer.build_inputs_with_special_tokens(seq)
    
    def save(self, filename):
        sequences = list(self.build())
        np.save(filename, np.array(sequences, dtype=np.int32))


# Version 1. Sequential processing, slow.
class DatasetBuilder:
    def __init__(self, tokenizer, corpus_dir, sequence_length=512, drop_last=True):
        self.tokenizer = tokenizer
        self.reader = CorpusReader(corpus_dir)
        self.packer = SequencePacker(sequence_length)
        self.drop_last = drop_last
        self.stats = BuildStats()

    def build(self):
        progress = tqdm(unit=" pieces")
        
        for doc in self.reader.documents():
            ids = self.tokenizer.encode(doc)
            self.stats.lines += 1
            self.stats.pieces += len(ids)
            progress.update(len(ids))
            for seq in self.packer.add(ids):
                seq = self.tokenizer.build_inputs_with_special_tokens(seq)
                self.stats.sequences += 1
                yield seq            

        for seq in self.packer.flush(drop_last=self.drop_last, pad_id=self.tokenizer.pad_token_id):
            seq = self.tokenizer.build_inputs_with_special_tokens(seq)
            self.stats.sequences += 1
            yield seq

        progress.close()            

    def save(self, filename):
        sequences = list(self.build())
        np.save(filename, np.array(sequences, dtype=np.int32))



@dataclass
class BuildStats:
    lines: int = 0
    pieces: int = 0
    sequences: int = 0
