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

# build and save the vocab
# corpus_dir = "20231101_vie"
# reader = CorpusReader(corpus_dir)
# builder = VocabularyBuilder(pipeline)
# vocab = builder.build(reader)
# vocab.save("{}.json".format(corpus_dir))

vocab = Vocabulary.load_and_prune("20231101_vie.json", 8)
print("vocab_size = {}".format(len(vocab)))
vocab.save("vocab.json")

corpus_dir = "0"
builder = DatasetBuilder(pipeline, vocab, "{}".format(corpus_dir), max_length=512)
builder.save("{}.npy".format(corpus_dir))
print("   pieces = {}".format(builder.stats.pieces))
print("    lines = {}".format(builder.stats.lines))
