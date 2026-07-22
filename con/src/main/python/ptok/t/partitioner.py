import os

class BytePartitioner:
    """
    Partition a large file into chunks, each chunk is defined by (start, end) offsets in bytes. 
    These offsets are used by CorpusShard to read multiple shards.
    """

    def partition(self, filename, num_workers=8):
        filesize = os.path.getsize(filename)
        chunk = filesize // num_workers
        offsets = [0]
        with open(filename, "rb") as fp:
            # find the beginning of each partition
            for i in range(1, num_workers):
                pos = i * chunk
                fp.seek(pos)
                # skip the remainder of the current line
                fp.readline()
                offsets.append(fp.tell())

        offsets.append(filesize)
        partitions = []
        for i in range(num_workers):
            partitions.append((offsets[i], offsets[i+1]))

        return partitions
    
