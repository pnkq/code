import sys
sys.path
sys.path.append('/home/phuonglh/code/con/src/main/python/ptok/')

from p.tokenizer import HybridTokenizer
from p.pipeline import Pipeline
from p.vocabulary import VocabularyBuilder, Vocabulary
from t.collator import MaskedLanguageModelDataCollator


text = "Tôi yêu ChatGPT."

pipeline = Pipeline()
vocab = Vocabulary.load("vocab.json")
tokenizer = HybridTokenizer(pipeline, vocab)

ids = tokenizer.encode(text)
ids = tokenizer.build_inputs_with_special_tokens(ids)

dataset = [
    {
        "input_ids": ids
    }
]

collator = MaskedLanguageModelDataCollator(tokenizer, debug=True)

batch = collator(dataset)

print(batch["input_ids"])
print(batch["labels"])
print(batch["attention_mask"])


# TEST 2: Visualize masking

def visualize(tokenizer, batch):
    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()

    print(f"{'Pos':>3} {'Input':>20} {'Label':>20}")

    for i, (x, y) in enumerate(zip(input_ids, labels)):
        input_tok = tokenizer.convert_ids_to_tokens(x)

        if y == -100:
            label_tok = "-"
        else:
            label_tok = tokenizer.convert_ids_to_tokens(y)

        print(f"{i:3d} {str(input_tok):>20} {str(label_tok):>20}")


visualize(tokenizer, batch)

# TEST 3: Statistics

eligible = 0
masked = 0

for _ in range(5000):
    batch = collator(dataset)

    eligible += (~batch["special_tokens_mask"]).sum().item()
    masked += batch["masked_indices"].sum().item()

print(masked / eligible)

num_tokens = 0
num_masked = 0


# TEST 4: 80/10/10 RULE

replace_count = 0
random_count = 0
unchanged_count = 0
masked_count = 0

for _ in range(5000):

    batch = collator(dataset)

    masked = batch["masked_indices"]
    replaced = batch["indices_replaced"]
    random = batch["indices_random"]

    n_masked = masked.sum().item()
    n_replaced = replaced.sum().item()
    n_random = random.sum().item()
    n_unchanged = n_masked - n_replaced - n_random

    masked_count += n_masked
    replace_count += n_replaced
    random_count += n_random
    unchanged_count += n_unchanged

print(f"Masked    : {masked_count}")
print(f"Mask      : {replace_count / masked_count:.4f}")
print(f"Random    : {random_count / masked_count:.4f}")
print(f"Unchanged : {unchanged_count / masked_count:.4f}")
print(f"Total     : {(replace_count + random_count + unchanged_count) / masked_count:.4f}")


# TEST 5: Special tokens must never be masked

for _ in range(1000):
    batch = collator(dataset)

    inp = batch["input_ids"][0]
    lab = batch["labels"][0]

    for token, label in zip(inp, lab):
        if token in { tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id }:
            assert label == -100


