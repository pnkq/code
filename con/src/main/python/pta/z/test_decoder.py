import sys
from pathlib import Path
import unittest

HOME_DIR = Path.home()
sys.path.append(f"{HOME_DIR}/code/con/src/main/python/ptok/")

from p.pipeline import Pipeline
from p.decoder import Decoder

class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline()
        self.decoder = Decoder()

    def roundtrip(self, text):
        pieces = self.pipeline.tokenize(text)
        decoded = self.decoder.decode(pieces)
        self.assertEqual(decoded, text)

    def test_vietnamese(self):
        self.roundtrip("Tôi đang học.")

    def test_english(self):
        self.roundtrip("I love transformers.")

    def test_mixed(self):
        self.roundtrip("Tôi thích OpenAI và transformers.")

if __name__ == "__main__":
    unittest.main()