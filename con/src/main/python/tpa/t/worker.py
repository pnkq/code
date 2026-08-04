from t.packer import SequencePacker
import numpy as np
from p.tokenizer import HybridTokenizer
from p.vocabulary import Vocabulary
from p.pipeline import Pipeline
from t.memmap import MemMapWriter
from t.stats import BuildStats

class WorkerBuilder:
    """
    Each worker should create its own tokenizer.
    """
    def __init__(self, id, shard, output_file, sequence_length, drop_last, progress_counter):
        self.id = id
        self.shard = shard
        self.output_file = output_file
        self.tokenizer = HybridTokenizer(Pipeline(), Vocabulary.load("vocab.json"))
        self.packer = SequencePacker(sequence_length)
        self.sequence_length = sequence_length
        self.drop_last = drop_last
        self.stats = BuildStats()
        self.progress_counter=progress_counter

    def build(self):
        # progress = tqdm(desc=f"Worker {self.id}", unit=" tokens", position=self.id, leave=True)
        pending = 0
        with MemMapWriter(self.output_file, self.sequence_length + 2) as writer:
            for line in self.shard.documents():
                before = self.packer.tokens_processed
                ids = self.tokenizer.encode_text(line)
                for seq in self.packer.add(ids):
                    writer.write(self._finalize(seq))
                count = self.packer.tokens_processed - before
                self.stats.lines += 1
                self.stats.pieces += count
                pending += count
                if pending >= 10000:
                    with self.progress_counter.get_lock():
                        self.progress_counter.value += pending
                    pending = 0

            if pending:
                with self.progress_counter.get_lock():
                    self.progress_counter.value += pending

            for seq in self.packer.flush(drop_last=self.drop_last, pad_id=self.tokenizer.pad_token_id):
                writer.write(self._finalize(seq)) # write sequence immediately

    def _finalize(self, seq):
        self.stats.sequences += 1
        return self.tokenizer.build_inputs_with_special_tokens(seq)
    

