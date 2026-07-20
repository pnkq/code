from torch.utils.data import Dataset
import numpy as np
import os


class MemMapWriter:
    def __init__(self, filename, sequence_length, dtype=np.int32, buffer_size=10000):
        self.filename = filename
        self.sequence_length = sequence_length
        self.dtype = np.dtype(dtype)
        self.buffer_size = buffer_size

        self.buffer = []
        self.count = 0

        self.fp = open(filename, "wb")

    def write(self, sequence):
        if len(sequence) != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {len(sequence)}."
            )

        self.buffer.append(sequence)
        self.count += 1

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return

        array = np.asarray(self.buffer, dtype=self.dtype)
        array.tofile(self.fp)
        self.buffer.clear()

    def close(self):
        self.flush()
        self.fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()



class MemMapDataset(Dataset):
    def __init__(self, filename, sequence_length, dtype=np.int32):
        self.filename = filename
        self.sequence_length = sequence_length
        self.dtype = np.dtype(dtype)

        itemsize = self.dtype.itemsize
        filesize = os.path.getsize(filename)

        record_size = sequence_length * self.dtype.itemsize

        if filesize % record_size != 0:
            raise ValueError(
                f"Corrupted memmap file.\n"
                f"File size = {filesize}\n"
                f"Record size = {record_size}"
            )

        self.num_sequences = filesize // record_size
        self.data = np.memmap(filename, mode="r", dtype=self.dtype, shape=(self.num_sequences, sequence_length))

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        return {
            "input_ids": np.array(self.data[idx], copy=True)
        }
    
