from corpus import CorpusReader
from packer import SequencePacker
from tqdm import tqdm
from stats import BuildStats

class DatasetBuilder:

    def __init__(self, tokenizer, corpus_dir, sequence_length=32, drop_last=True):
        self.tokenizer = tokenizer
        self.reader = CorpusReader(corpus_dir)
        self.packer = SequencePacker(sequence_length)
        self.drop_last = drop_last
        self.stats = BuildStats()

    def build(self):
        progress = tqdm(unit=" tokens")
        pending = 0
        for line in self.reader.transitions(): # use documents() for text
            before = self.packer.tokens_processed
            ids = self.tokenizer.encode(line)
            for seq in self.packer.add(ids):
                yield self._finalize(seq)
            count = self.packer.tokens_processed - before
            self.stats.lines += 1
            self.stats.pieces += count
            pending += count
            if pending >= 5000:
                progress.update(pending)
                pending = 0

        if pending > 0:
            progress.update(pending)

        for seq in self.packer.flush(drop_last=self.drop_last, pad_id=self.tokenizer.pad_token_id):
            yield self._finalize(seq)

        progress.close()

    def _finalize(self, seq):
        self.stats.sequences += 1
        return self.tokenizer.build_inputs_with_special_tokens(seq)
    

