from pathlib import Path
from tqdm import tqdm 

class CorpusReader:
    def __init__(self, corpus_dir):
        self.corpus_dir = Path(corpus_dir)

    def documents(self):
        progress = tqdm(unit=" lines")
        n = 2000
        i = 0
        for file in sorted(self.corpus_dir.glob("*.txt")):
            with open(file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        i += 1
                        yield line
                    if i % n == 0:
                        progress.update(n)
                        i = 0
        if i > 0:
            progress.update(i)
            
        progress.close()

