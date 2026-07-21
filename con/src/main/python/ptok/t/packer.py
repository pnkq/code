
class SequencePacker:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.buffer = [0] * sequence_length # use a fixed-size, preallocated list to avoid list copy
        self.size = 0
        self.tokens_processed = 0

    def add(self, ids):
        for id in ids:
            self.buffer[self.size] = id
            self.size += 1
            self.tokens_processed += 1

            if self.size == self.sequence_length:
                yield self.buffer.copy()
                self.size = 0

    def flush(self, drop_last=True, pad_id=None):
        if self.size == 0:
            return
        if drop_last:
            self.size = 0
            return
        if pad_id is None:
            raise ValueError("pad_id must be specified when drop_last=False")
        sequence = self.buffer[:self.size]
        sequence.extend([pad_id] * (self.sequence_length - self.size))
        self.size = 0
        yield sequence


