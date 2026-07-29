import gzip
import json
import sys
from typing import Iterator

# Check if tqdm is installed, otherwise fallback to a dummy wrapper
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: 'tqdm' module not found. Running without progress bar.")
    print("To install it, run: pip install tqdm\n")

    def tqdm(iterable, *args, **kwargs):
        return iterable


def read_jsonl_gz_texts(file_path: str) -> Iterator[str]:
    """Reads a .jsonl.gz file and yields only the 'text' field."""
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                yield data["text"]


def write_texts_to_file(texts: Iterator[str], output_path: str, total_lines: int = None) -> None:
    """Writes an iterator of strings to a text file with a progress bar."""
    with open(output_path, "w", encoding="utf-8") as f:
        # Wrap the iterator with tqdm for visual feedback
        for text in tqdm(texts, total=total_lines, desc="Processing lines", unit=" lines"):
            f.write(text.rstrip("\n") + "\n")


def main(input_file: str, output_file: str) -> None:
    """Chains the extraction and writing processes together."""
    print(f"Starting extraction from {input_file}...")

    # Initialize the iterator
    text_iterator = read_jsonl_gz_texts(input_file)

    # Process and write data
    write_texts_to_file(text_iterator, output_file)

    print(f"Successfully saved all texts to {output_file}")


if __name__ == "__main__":
    # Example paths - replace with your actual file paths
    INPUT_GZ_FILE = "c4_vi_validation.jsonl.gz"
    OUTPUT_TXT_FILE = "c4_vi_validation.txt"

    main(INPUT_GZ_FILE, OUTPUT_TXT_FILE)
