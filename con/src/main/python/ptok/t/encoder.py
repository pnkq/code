class CorpusEncoder:
    def __init__(self, pipeline, vocab):
        self.pipeline = pipeline
        self.vocab = vocab

    def encode_document(self, text):
        # Piece stream
        pieces = self.pipeline.tokenize(text)
        # ids
        ids = self.vocab.encode(pieces)
        return ids
    
