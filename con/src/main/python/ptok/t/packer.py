class SequencePacker:
    def __init__(self, max_length):
        self.max_length = max_length
        self.buffer = []

    def add(self, ids):
        self.buffer.extend(ids)
        while len(self.buffer) >= self.max_length:
            yield self.buffer[:self.max_length]
            self.buffer = self.buffer[self.max_length:]

    # drop the last incomplete sequence
    def flush(self):
        self.buffer = []
        return []
    



