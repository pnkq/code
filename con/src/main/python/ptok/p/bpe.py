from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFC

# Create tokenizer, don't use ByteLevel()
tokenizer = Tokenizer(BPE())

# Unicode normalization: Composed Unicode (NFC - Normalization Form Canonical Composition)
tokenizer.normalizer = NFC()

# IMPORTANT:
# Split only on whitespace.
# No ByteLevel pre-tokenizer.
tokenizer.pre_tokenizer = Whitespace()

# Trainer

trainer = BpeTrainer(vocab_size=16384, min_frequency=2, special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"])

tokenizer.train(["corpus_2_eng.txt"], trainer)

# Save
tokenizer.save("bpe_eng.json")

print("Vocabulary size:", tokenizer.get_vocab_size())