import torch
import torch.nn as nn

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 1. Preprocessing pipeline
import torchtext.transforms as T
from torch.hub import load_state_dict_from_url

padding_idx = 1
bos_idx = 0
eos_idx = 2
max_seq_len = 256
xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

text_transform = T.Sequential(
    T.SentencePieceTokenizer(xlmr_spm_model_path),
    T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    T.Truncate(max_seq_len - 2),
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
)

# text_transform = XLMR_BASE_ENCODER.transform()


# 2. Data pipeline
from torch.utils.data import DataLoader

from torchtext.datasets import SST2

batch_size = 16

train_datapipe = SST2(split="train")
dev_datapipe = SST2(split="dev")

def batch_transform(x):
    return {"token_ids": text_transform(x["text"]), "target": x["label"]}


train_datapipe = train_datapipe.batch(batch_size).rows2columnar(["text", "label"])
train_datapipe = train_datapipe.map(batch_transform)
dev_datapipe = dev_datapipe.batch(batch_size).rows2columnar(["text", "label"])
dev_datapipe = dev_datapipe.map(batch_transform)

train_dataloader = DataLoader(train_datapipe, batch_size=None)
dev_dataloader = DataLoader(dev_datapipe, batch_size=None)

first = next(iter(dev_dataloader))
print(first)

# 3. Model definition
num_classes = 2
input_dim = 768

from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER

classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=input_dim)
model = XLMR_BASE_ENCODER.get_model(head=classifier_head)
model.to(DEVICE)

# 4. Training
import torchtext.functional as F
from torch.optim import AdamW

learning_rate = 1e-5
optim = AdamW(model.parameters(), lr=learning_rate)
criteria = nn.CrossEntropyLoss()


def train_step(input, target):
    output = model(input)
    loss = criteria(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()


def eval_step(input, target):
    output = model(input)
    loss = criteria(output, target).item()
    return float(loss), (output.argmax(1) == target).type(torch.float).sum().item()


def evaluate():
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    counter = 0
    with torch.no_grad():
        for batch in dev_dataloader:
            input = F.to_tensor(batch["token_ids"], padding_value=padding_idx).to(DEVICE)
            target = torch.tensor(batch["target"]).to(DEVICE)
            loss, predictions = eval_step(input, target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1

    return total_loss / counter, correct_predictions / total_predictions

num_epochs = 10

for e in range(num_epochs):
    for batch in train_dataloader:
        input = F.to_tensor(batch["token_ids"], padding_value=padding_idx).to(DEVICE)
        target = torch.tensor(batch["target"]).to(DEVICE)
        train_step(input, target)
    loss, accuracy = evaluate()
    print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(e, loss, accuracy))
