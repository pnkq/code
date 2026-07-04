from p.piece import Piece


class PostProcessor:
    def process(self, pieces):
        output = []
        output.append(Piece(text="<s>", source="special", language="special", start=-1, end=-1))
        output.extend(pieces)
        output.append(Piece(text="</s>", source="special", language="special", start=-1, end=-1))
        return output
    
