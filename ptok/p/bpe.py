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

trainer = BpeTrainer(vocab_size=16384, min_frequency=5, special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"])

#tokenizer.train(["20231101/part-00000-b2514431-1ed1-4374-ba72-5814e6ba27cf-c000.txt"], trainer)
tokenizer.train(["20231101/eng.txt"], trainer)

# Save
#tokenizer.save("p/bpe_vie.json")
tokenizer.save("p/bpe_eng.json")

print("Vocab size:", tokenizer.get_vocab_size())