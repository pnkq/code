import unicodedata

class Normalizer:
    def normalize(self, text):
        text = unicodedata.normalize("NFC", text)
        text = " ".join(text.split())
        return text

