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
corpus_dir = "0"
reader = CorpusReader(corpus_dir)
builder = VocabularyBuilder(pipeline)
vocab = builder.build(reader)
vocab.save("{}.json".format(corpus_dir))

# vocab = Vocabulary.load("{}.json".format(corpus_dir))

builder = DatasetBuilder(pipeline, vocab, "{}".format(corpus_dir), max_length=512)
builder.save("{}.npy".format(corpus_dir))
print("   pieces = {}".format(builder.stats.pieces))
print("documents = {}".format(builder.stats.documents))
