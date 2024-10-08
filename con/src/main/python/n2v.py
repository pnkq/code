import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec


import csv
def read(path):
    with open(path) as file:
        reader = csv.reader(file, delimiter=" ")
        lines = []
        for i, line in enumerate(reader):
            lines.append([int(u) for u in line])
    edge_index = torch.tensor([lines[0], lines[1]], dtype=torch.long)
    y = torch.tensor(lines[2], dtype=torch.long)
    # N = len(lines[2])
    # # node features, all is [0.]
    # x = torch.tensor([torch.zeros(1) for _ in range(N)], dtype=torch.float).reshape(N, 1)
    data = Data(edge_index=edge_index, y=y)
    return data

basePath = "/home/phuonglh/code/con/dat/dep"
language = "fra"

data = read(f"{basePath}/{language}-pyg.tsv")
print(data)
print(data.keys())
print(data.num_node_features)
print(data.num_nodes)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Node2Vec(
    data.edge_index,
    embedding_dim=32,
    walks_per_node=10,
    walk_length=20,
    context_size=10,
    p=1.0,
    q=1.0,
    num_negative_samples=1,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loader = model.loader(batch_size=128, shuffle=True, num_workers=4)

def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


z = model()  # Full node-level embeddings.
z = model(torch.tensor([0, 1, 2]))  # Embeddings of first three nodes.
print(z)

# write out embeddings file (this is paired with {lang}-nodeId.txt file)
embeddings = model(torch.tensor(range(data.num_nodes)))
with open(f'{basePath}/{language}-nodeVec.txt', 'w') as f:
    es = embeddings.tolist()
    for e in es:
        s = ' '.join(map(str, e))
        f.write(s)
        f.write('\n')