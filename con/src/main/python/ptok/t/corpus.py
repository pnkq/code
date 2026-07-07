from pathlib import Path


class CorpusReader:
    def __init__(self, corpus_dir):
        self.corpus_dir = Path(corpus_dir)

    def documents(self):
        for file in sorted(self.corpus_dir.glob("*.txt")):
            with open(file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line