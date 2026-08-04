from pathlib import Path
from tqdm import tqdm 

# (C) phuonglh@gmail.com

class CorpusReader:
    """
    Read all *.txt files in a directory into an iterator of lines.
    """
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


class CorpusShard:
    """
    A corpus shard that reads all bytes in a particular range (start, end).
    This is the mechanism that allows sharding a very large file into multiple partitions.
    """
    def __init__(self, filename, start, end):
        self.filename = filename
        self.start = start
        self.end = end

    def documents(self):
        with open(self.filename, "rb") as fp:
            fp.seek(self.start)
            while fp.tell() < self.end:
                line = fp.readline()
                if not line:
                    break
                yield line.decode("utf-8").rstrip("\n")

