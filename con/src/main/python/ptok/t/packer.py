class SequencePacker:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.buffer = []

    def add(self, ids):
        self.buffer.extend(ids)
        while len(self.buffer) >= self.sequence_length:
            sequence = self.buffer[:self.sequence_length]
            self.buffer = self.buffer[self.sequence_length:]
            yield sequence

    def flush(self, drop_last=True, pad_id=None):
        if not self.buffer:
            return
        if drop_last:
            self.buffer.clear()
            return
        if pad_id is None:
            raise ValueError("pad_id must be specified when drop_last=False")
        sequence = self.buffer + [ pad_id ] * (self.sequence_length - len(self.buffer))
        self.buffer.clear()
        yield sequence    

