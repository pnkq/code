from itertools import groupby

class Decoder:

    def decode(self, pieces):
        pieces = [p for p in pieces if p.language != 'pad']

        # They should already be ordered, but this makes the
        # decoder robust.
        pieces.sort(key=lambda p: p.start)

        result = []
        cursor = 0

        for piece in pieces:
            # Fill the gap with spaces
            while cursor < piece.start:
                result.append(" ")
                cursor += 1
            result.append(piece.text)
            cursor = piece.end

        return "".join(result)
