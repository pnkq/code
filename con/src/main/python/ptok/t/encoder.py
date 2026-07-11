from p.tokenizer import HybridTokenizer

class CorpusEncoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text)
    
    
    
