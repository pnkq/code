from p.vocabulary import VocabularyBuilder, Vocabulary
from p.pipeline import Pipeline
from t.dataset import CorpusReader, DatasetBuilder, DatasetBuilderPar
from p.tokenizer import HybridTokenizer
from t.memmap import MemMapWriter, MemMapDataset
from t.partitioner import BytePartitioner

import argparse
import sys

def tokenize_simple_text(pipeline):
    """Test 1: Tokenize a simple text."""
    text = """Tôi đang học transformers của OpenAI và SpaceX."""
    pieces = pipeline.tokenize(text)
    for p in pieces:
        print(p)
    
    print()
    subs = pipeline.tokenize_to_text(text)
    for s in subs:
        print(s)

def build_vocabulary(corpus_dir):
    """Test 2: Build and save a vocabulary."""
    reader = CorpusReader(corpus_dir)
    pipeline = Pipeline()
    builder = VocabularyBuilder(pipeline)
    vocab = builder.build(reader)
    vocab.save("{}.json".format(corpus_dir))

def prune_vocabulary(input_vocab_file, output_vocab_file="vocab.json", min_frequency=8):
    """Test 3: Load and prune a vocabulary, reindex entry ids; the output is 'vocab.json'."""
    vocab = Vocabulary.load_and_prune(input_vocab_file, min_frequency)
    print(f"vocab_size = {len(vocab)}")
    vocab.save(output_vocab_file)

def tokenize(pipeline, vocab_file="vocab.json"):
    """Test 4a: Test a hybrid tokenizer using a pre-built vocabulary."""
    vocab = Vocabulary.load(vocab_file)
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

def tokenize_text(pipeline, vocab_file="vocab.json"):
    """Test 4b: Test a hybrid tokenizer using a pre-built vocabulary."""
    vocab = Vocabulary.load(vocab_file)
    tokenizer = HybridTokenizer(pipeline, vocab)
    text = """Tôi đang học transformers của OpenAI."""
    result = tokenizer.encode_text(text)
    print(result)
    print()

def dataset_builder(tokenizer, corpus_dir):
    """Test 5: Convert a corpus into sequences of ids."""
    builder = DatasetBuilder(tokenizer, "{}".format(corpus_dir), sequence_length=32)
    seqs = builder.build()
    for s in seqs:
        print(s)
    print(f"   pieces = {builder.stats.pieces}")
    print(f"    lines = {builder.stats.lines}")
    print(f"sequences = {builder.stats.sequences}")

def memmap_writer(tokenizer, corpus_dir, sequence_length):
    """Test 6: Save the corpus into a *.bin file for training, then load the binary dataset into a mem-map dataset."""
    builder = DatasetBuilder(tokenizer, f"{corpus_dir}", sequence_length=sequence_length)
    writer = MemMapWriter(f"{corpus_dir}.bin", sequence_length=sequence_length+2)

    # trigger the generator...
    for seq in builder.build():
        writer.write(seq)

    writer.close()
    
def memmap_dataset(corpus_dir, sequence_length):
    dataset = MemMapDataset(f"{corpus_dir}.bin",  sequence_length=sequence_length+2)
    print(f"Number of sequences = {len(dataset)}")
    sample = dataset[0]
    print(f'Shape of a sequence is {sample["input_ids"].shape}')
        

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Run a specific function based on a command-line argument."
    )
    
    # Define the choices available to the user
    parser.add_argument(
        'action',
        choices=['tokenize', 'vocab', 'prune', 'memmap', 'dataset', 'partition', "memmap_par"],
        help="The specific function/action you want to execute."
    )

    # Parse the arguments from command line.
    args = parser.parse_args()
    pipeline = Pipeline()

    match args.action:
        case 'tokenize': 
            tokenize_simple_text(pipeline)
            tokenize(pipeline)
            tokenize_text(pipeline)
        case 'vocab': 
            build_vocabulary("20231101_vie")
        case 'prune':
            prune_vocabulary("20231101_vie", "vocab.json")
        case 'memmap': 
            tokenizer = HybridTokenizer(pipeline, Vocabulary.load("vocab.json"))
            memmap_writer(tokenizer, "1", 510)
            memmap_dataset("1", 510)
            # memmap_writer(tokenizer, "2", 510)
            # memmap_dataset("2", 510)
            # memmap_writer(tokenizer, "20231101_vie", 510)
            # memmap_dataset("20231101_vie", 510)
        case 'dataset': 
            tokenizer = HybridTokenizer(pipeline, Vocabulary.load("vocab.json"))
            dataset_builder(tokenizer, "0")
        case 'partition':
            # this is a big text file (1.3GB):
            filename = "/home/phuonglh/code/con/src/main/python/ptok/20231101_vie/part-00000-b2514431-1ed1-4374-ba72-5814e6ba27cf-c000.txt"
            offsets = BytePartitioner().partition(filename, num_workers=8)
            for pair in offsets:
                print(pair)
        case 'memmap_par':
            corpus_file = "2/corpus_2.txt"
            builder = DatasetBuilderPar(sequence_length=510, num_workers=8)
            builder.build(corpus_file, "2.bin")
            # corpus_file = "/home/phuonglh/corpora/oscar/21/vi_part_1.txt"
            # builder = DatasetBuilderPar(sequence_length=510, num_workers=16)
            # builder.build(corpus_file, "vi_part_1.bin")
            # corpus_file = "/home/phuonglh/code/con/src/main/python/ptok/20231101_vie/part-00000-b2514431-1ed1-4374-ba72-5814e6ba27cf-c000.txt"
            # builder = DatasetBuilderPar(sequence_length=510, num_workers=10)
            # builder.build(corpus_file, "v.bin")
        case _:
            print("Invalid action selection.", file=sys.stderr)

if __name__ == "__main__":
    main()


