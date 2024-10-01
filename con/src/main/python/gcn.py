import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# dataset = Planetoid(root='/home/phuonglh/code/con/dat/pyg/', name='Cora')
import csv
def read(path):
    with open(path) as file:
        reader = csv.reader(file, delimiter=" ")
        lines = []
        for i, line in enumerate(reader):
            lines.append([int(u) for u in line])
    edge_index = torch.tensor([lines[0], lines[1]], dtype=torch.long)
    y = torch.tensor(lines[2], dtype=torch.long)
    N = len(lines[2])
    # node features, all is [0.]
    x = torch.tensor([torch.zeros(1) for _ in range(N)], dtype=torch.float).reshape(N, 1)
    mask = torch.ones(N, dtype=torch.bool)
    data = Data(edge_index=edge_index, y=y, x=x, train_mask=mask, val_mask=mask, test_mask=mask)
    return data

data = read("/home/phuonglh/code/con/dat/dep/eng-pyg.tsv")
print(data)
print(data.keys())
print(data.num_node_features)

num_classes = 18

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(100):
    print('epoch = {}'.format(epoch))
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

