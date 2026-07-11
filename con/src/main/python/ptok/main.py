import unicodedata
from p.vocabulary import VocabularyBuilder, Vocabulary
from p.pipeline import Pipeline
from t.dataset import DatasetBuilder
from p.tokenizer import HybridTokenizer


pipeline = Pipeline()

# TEST 1: tokenize a simple text
# text = """Tôi đang học transformers của OpenAI và SpaceX."""
# pieces = pipeline.tokenize(text)
# for p in pieces:
#     print(p)

# TEST 2: build and save a vocab
# corpus_dir = "20231101_vie"
# reader = CorpusReader(corpus_dir)
# pipeline = Pipeline()
# builder = VocabularyBuilder(pipeline)
# vocab = builder.build(reader)
# vocab.save("{}.json".format(corpus_dir))

# TEST 3: load and prune a vocab
# vocab = Vocabulary.load_and_prune("20231101_vie.json", 8)
# print("vocab_size = {}".format(len(vocab)))
# vocab.save("vocab.json")

# TEST 4: test the HybridTokenizer
vocab = Vocabulary.load("vocab.json")
tokenizer = HybridTokenizer(pipeline, vocab)
text = """Tôi đang học transformers của OpenAI."""
result = tokenizer.encode(text)
print(result)
print()
result2 = tokenizer(text)
print(result2)
print()
result3 = tokenizer([text, "Thực hiện theo ý kiến chỉ đạo của Giám đốc Đại học Quốc gia Hà Nội"])
print(result3)

# TEST 5: test the dataset builder that convert a corpus into sequences of ids
corpus_dir = "0"
builder = DatasetBuilder(tokenizer, "{}".format(corpus_dir), sequence_length=32)
seqs = builder.build()
for s in seqs:
    print(s)
# builder.save("{}.npy".format(corpus_dir))
print("   pieces = {}".format(builder.stats.pieces))
print("    lines = {}".format(builder.stats.lines))
print(" sequences = {}".format(builder.stats.sequences))
