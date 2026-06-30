class Vocabulary:
    def __init__(self):
        self.token2id = {}
        self.id2token = {}

    def add(self, token):
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

        return self.token2id[token]
    
    def encode(self, pieces):
        return [self.add(x.text) for x in pieces]
    
