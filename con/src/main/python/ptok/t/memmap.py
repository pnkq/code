from torch.utils.data import Dataset
import numpy as np
import torch


class MemMapWriter:
    def __init__(self, filename, dtype=np.int32):
        self.filename = filename
        self.dtype = dtype
        self._buffer = []
        self.count = 0

    def write(self, ids):
        self._buffer.append(ids)
        self.count += 1

    def close(self):
        array = np.asarray(self._buffer, dtype=self.dtype)
        np.save(self.filename, array)



class MemMapDataset(Dataset):
    def __init__(self, filename):
        self.data = np.load(filename, mmap_mode="r")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.data[idx], dtype=torch.long)
        }
    
