from pathlib import Path
from tqdm import tqdm 

class CorpusReader:
    def __init__(self, corpus_dir):
        self.corpus_dir = Path(corpus_dir)

    def documents(self):
        progress = tqdm(unit=" lines")
        for file in sorted(self.corpus_dir.glob("*.txt")):
            with open(file, encoding="utf-8") as f:
                i = 0
                for line in f:
                    line = line.strip()
                    if line:
                        i += 1
                        yield line
                    if i % 1000 == 0:
                        progress.update(1000)
        progress.close()

