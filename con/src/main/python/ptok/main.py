import unicodedata
from p.tokenizer import HybridTokenizer
from p.vocabulary import VocabularyBuilder
from p.pipeline import Pipeline


pipeline = Pipeline()

text = """Tôi đang học transformers của OpenAI và SpaceX."""

pieces = pipeline.tokenize(text)

for p in pieces:
    print(p)


# vocab_builder = VocabularyBuilder()
# vocab_builder.add_stream(pieces)
# vocab = vocab_builder.build()
# vocab_builder.save(vocab, "vocab.json", "stat.json")


