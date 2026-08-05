from vocabulary import VocabularyBuilder, Vocabulary
from dataset import CorpusReader, DatasetBuilder
from tokenizer import TransitionTokenizer
from memmap import MemMapWriter, MemMapDataset
from e.intrinsic import MLMAccuracyEvaluator

import argparse
import sys

def build_vocabulary(corpus_dir):
    """Build and save a vocabulary."""
    reader = CorpusReader(corpus_dir)
    builder = VocabularyBuilder()
    vocab = builder.build(reader)
    vocab.save("vocab.json")

def tokenize():
    """Test a tokenizer using a pre-built vocabulary."""
    vocab = Vocabulary.load("vocab.json")
    tokenizer = TransitionTokenizer(vocab)
    result = tokenizer(["SH RA-cop SH SH LA-det SH SH LA-det", "LA-case SH RA-compound"])
    print(result)

def dataset_builder(tokenizer, corpus_dir, sequence_length=32):
    """Convert a transition corpus into sequences of ids."""
    builder = DatasetBuilder(tokenizer, f"{corpus_dir}", sequence_length=sequence_length)
    seqs = builder.build()
    for s in seqs:
        print(s)
    print(f"transitions = {builder.stats.pieces}")
    print(f"      lines = {builder.stats.lines}")
    print(f"  sequences = {builder.stats.sequences}")

def memmap_writer(tokenizer, corpus_dir, sequence_length):
    """Save the corpus into a *.bin file for training, then load the binary dataset into a mem-map dataset."""
    builder = DatasetBuilder(tokenizer, f"{corpus_dir}", sequence_length=sequence_length)
    writer = MemMapWriter(f"{corpus_dir}/{corpus_dir}.bin", sequence_length=sequence_length+2)

    # trigger the generator...
    for seq in builder.build():
        writer.write(seq)

    writer.close()
    
def memmap_dataset(corpus_dir, sequence_length):
    dataset = MemMapDataset(f"{corpus_dir}/{corpus_dir}.bin",  sequence_length=sequence_length+2)
    print(f"Number of sequences = {len(dataset)}")
    sample = dataset[0]
    print(f'Shape of a sequence is {sample["input_ids"].shape}')

def evaluate(corpus_dir, tokenizer, model):
    from torch.utils.data import DataLoader
    from collator import MaskedLanguageModelDataCollator

    collator = MaskedLanguageModelDataCollator(tokenizer, mlm_probability=0.15, debug=True)
    dataset = MemMapDataset(f"{corpus_dir}/{corpus_dir}.bin", sequence_length=32+2)
    print(f"Number of sequences = {len(dataset)}")
    sample = dataset[0]
    print(f'Shape of a sequence is {sample["input_ids"].shape}')
    print(sample)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collator)
    batch = next(iter(loader))
    print(batch.keys())
    print(batch["input_ids"].shape)
    print(batch["attention_mask"].shape)
    print(batch["labels"].shape)    
    labels = batch["labels"][0]

    masked_positions = (labels != -100).nonzero(as_tuple=True)[0]
    print(masked_positions)

    for pos in masked_positions:
        print(
            pos.item(),
            tokenizer.convert_ids_to_tokens([batch["input_ids"][0][pos].item()]),
            tokenizer.decode([labels[pos].item()])
        )
    # evaluator = MLMAccuracyEvaluator(model, device="cpu", top_k=5)
    # result = evaluator.evaluate(loader)
    # print(result)

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Run a specific function based on a command-line argument."
    )
    
    # Define the choices available to the user
    parser.add_argument(
        'action',
        choices=['tokenize', 'vocab', 'prune', 'dataset', "memmap", "evaluate"],
        help="The specific function/action you want to execute."
    )

    # Parse the arguments from command line.
    args = parser.parse_args()

    match args.action:
        case 'vocab': 
            build_vocabulary("0")
        case 'tokenize': 
            tokenize()
        case 'dataset': 
            tokenizer = TransitionTokenizer(Vocabulary.load("vocab.json"))
            dataset_builder(tokenizer, "0", 16)
        case 'memmap':
            # tokenizer = TransitionTokenizer(Vocabulary.load("vocab.json"))
            # memmap_writer(tokenizer, "0", 32)
            memmap_dataset("0", 32)
        case 'evaluate':
            tokenizer = TransitionTokenizer(Vocabulary.load("vocab.json"))
            evaluate("0", tokenizer, None)
        case _:
            print("Invalid action selection.", file=sys.stderr)

if __name__ == "__main__":
    main()


