from torch.utils.data import Dataset, DataLoader
import numpy as np

class DummyDataset(Dataset):

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return {"input_ids": np.zeros(512, dtype=np.int32)}

loader = DataLoader(
    DummyDataset(),
    batch_size=8,
    num_workers=0
)

print("Before iterator")
it = iter(loader)
print("Iterator created")
batch = next(it)
print(batch["input_ids"].shape)

