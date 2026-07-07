import unicodedata
from p.vocabulary import VocabularyBuilder, Vocabulary
from p.pipeline import Pipeline
from t.dataset import DatasetBuilder
from t.corpus import CorpusReader


pipeline = Pipeline()

# text = """Tôi đang học transformers của OpenAI và SpaceX."""
# pieces = pipeline.tokenize(text)
# for p in pieces:
#     print(p)

# vocab_builder = VocabularyBuilder()
# vocab_builder.add_stream(pieces)
# vocab = vocab_builder.build()
# vocab_builder.save(vocab, "vocab.json", "stat.json")

# build and save the vocab
reader = CorpusReader("d")
builder = VocabularyBuilder(pipeline)
vocab = builder.build(reader)
vocab.save("d.json")

# vocab = Vocabulary.load("d.json")

builder = DatasetBuilder(pipeline, vocab, "d", max_length=512)
builder.save("0.npy")
print("   pieces = {}".format(builder.stats.pieces))
print("documents = {}".format(builder.stats.documents))
