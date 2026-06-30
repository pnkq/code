from transformers import AutoTokenizer

class EnglishTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize(self, word):
        return self.tokenizer.tokenize(word)