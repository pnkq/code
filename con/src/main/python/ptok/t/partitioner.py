import os

class BytePartitioner:
    """
    Partition a large file into chunks, each chunk is defined by (start, end) offsets in bytes. 
    These offsets are used by CorpusShard to read multiple shards.
    """

    def partition(self, filename, num_workers=8):
        filesize = os.path.getsize(filename)
        chunk = filesize // num_workers
        offsets = []
        for i in range(num_workers):
            start = i * chunk
            if i == num_workers - 1:
                end = filesize
            else:
                end = (i + 1) * chunk
            offsets.append((start, end))
        return offsets
    
