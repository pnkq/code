from p.piece import Piece


class PostProcessor:
    def __init__(self, max_length=512):
        self.max_length = max_length

    def process(self, pieces):
        out = pieces[:self.max_length - 2]
        output = []
        output.append(Piece(text="<s>", source="pad", language="pad", start=-1, end=-1))
        output.extend(out)
        output.append(Piece(text="</s>", source="pad", language="pad", start=-1, end=-1))
        return output
    
