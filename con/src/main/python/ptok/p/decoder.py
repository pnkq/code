from itertools import groupby

class Decoder:

    def decode(self, pieces):
        words = []
        pieces = [p for p in pieces if p.language != 'special']
        for _, group in groupby(pieces, key=lambda p: p.word_id):
            word = "".join(piece.text for piece in group)
            words.append(word)

        text = ""
        for word in words:
            if not text:
                text = word
            # elif word in PUNCT:
            #     text += word
            else:
                text += " " + word

        return text
