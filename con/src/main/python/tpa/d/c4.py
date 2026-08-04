from datasets import load_dataset
import json
import gzip

# streaming=True bypasses HF disk cache completely
split = "validation" # "validation"
output_file = f"c4_vi_{split}.jsonl.gz"

# 1. Load dataset stream
c4_stream = load_dataset("allenai/c4", "vi", split=split, streaming=True)

# 2. Write directly to gzip file in text mode ("wt")
print("Streaming and compressing data to .jsonl.gz...")
with gzip.open(output_file, "wt", encoding="utf-8") as f:
    for count, sample in enumerate(c4_stream, start=1):
        # Compress and write sample on-the-fly
        f.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")
        
        if count % 100000 == 0:
            print(f"Compressed and saved {count} samples...")

print(f"Finished! File saved to {output_file}")