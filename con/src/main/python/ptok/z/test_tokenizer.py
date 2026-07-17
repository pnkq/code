import sys
from pathlib import Path
import unittest

HOME_DIR = Path.home()
sys.path.append(f"{HOME_DIR}/code/con/src/main/python/ptok/")

from p.pipeline import Pipeline

class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline()

    def roundtrip(self, text):
        pieces = self.pipeline.tokenize(text)
        decoded = self.decoder.decode(pieces)
        self.assertEqual(decoded, text)

    def test_vietnamese(self):
        pieces = self.pipeline.tokenize("Tôi đang học.")
        output = [p.text for p in pieces]
        self.assertEqual(output, ["T", "ôi", "đ", "a", "ng", "h", "ọ", "c", "."])

    def test_english(self):
        pieces = self.pipeline.tokenize("I love transformers.")
        output = [p.text for p in pieces]
        self.assertEqual(output, ["I", "love", "transform", "ers", "."])

    def test_mixed(self):
        pieces = self.pipeline.tokenize("Tôi thích OpenAI và transformers.")
        output = [p.text for p in pieces]
        self.assertEqual(output, ["T", "ôi", "th", "í", "ch", "Open", "AI", "v", "à", "transform", "ers", "."])

if __name__ == "__main__":
    unittest.main()