import os
import shutil

class DatasetMerger:
    """
    Merge multiple binary files into one big binary file. 
    """
    def merge(self, parts, output):
        with open(output, "wb") as fout:
            for filename in parts:
                with open(filename, "rb") as fin:
                    shutil.copyfileobj(fin, fout)

        for part in parts:
            os.remove(part)
                    