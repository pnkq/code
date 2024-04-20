import torch
import torch.nn as nn

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 1. Preprocessing pipeline
import torchtext.transforms as T
from torch.hub import load_state_dict_from_url

from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER

padding_idx = 1
# bos_idx = 0
# eos_idx = 2
# max_seq_len = 256
# xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
# xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

# text_transform = T.Sequential(
#     T.SentencePieceTokenizer(xlmr_spm_model_path),
#     T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
#     T.Truncate(max_seq_len - 2),
#     T.AddToken(token=bos_idx, begin=True),
#     T.AddToken(token=eos_idx, begin=False),
# )

text_transform = XLMR_BASE_ENCODER.transform()

# get the home user
from os.path import expanduser
home = expanduser("~")
dsl = home + "/manuscripts/code/dsl"
data_base_path = dsl + "/dat/DSL-ML-2024"
model_base_path = dsl + "/bin"


import sys

language = sys.argv[1].strip() # first argument beside the script name
print("language = ", language)

# get the dataset
train_path = data_base_path + "/{}/{}_train.tsv".format(language, language)
dev_path = data_base_path + "/{}/{}_dev.tsv".format(language, language)
test_path = data_base_path + "/{}/{}_test.tsv".format(language, language)

# 2. Data pipeline
from torch.utils.data import DataLoader
import torchdata.datapipes as dp

train_datapipe = dp.iter.FileOpener([train_path], mode='rt')
train_datapipe = train_datapipe.parse_csv(delimiter='\t')
dev_datapipe = dp.iter.FileOpener([dev_path], mode='rt')
dev_datapipe = dev_datapipe.parse_csv(delimiter='\t')
test_datapipe = dp.iter.FileOpener([test_path], mode='rt')
test_datapipe = test_datapipe.parse_csv(delimiter='\t')

batch_size = 16

# EN
labels_dict = {"EN-GB": 0, "EN-US": 1, "EN-GB,EN-US": 2}

if language == "ES":
    labels_dict = {"ES-AR": 0, "ES-ES": 1, "ES-AR,ES-ES": 2}
elif language == "FR":
    labels_dict = {"FR-BE,FR-CA": 0, "FR-CA": 1, "FR-BE,FR-CH": 2, "FR-BE": 3, "FR-CH": 4, "FR-FR": 5, "FR-BE,FR-FR": 6,
    "FR-CA,FR-FR": 7, "FR-CH,FR-FR": 8, "FR-BE,FR-CH,FR-FR": 9}
elif language == "PT":
    labels_dict = {"PT-BR": 0, "PT-PT": 1, "PT-BR,PT-PT": 2}
elif language == "BCMS":
    labels_dict = {"me": 0, "sr": 1, "hr": 2, "bs": 3, "bs,hr": 4}

labels_map = {v: k for k, v in labels_dict.items()}

print(labels_dict)
print(labels_map)

def batch_transform(x):
    return {"token_ids": text_transform(x["text"]), "target": [labels_dict[k] for k in x["languages"]]}

def batch_transform_test(x):
    return {"token_ids": text_transform(x["text"])}

train_datapipe = train_datapipe.batch(batch_size).rows2columnar(["languages", "text"])
train_datapipe = train_datapipe.map(batch_transform)
dev_datapipe = dev_datapipe.batch(batch_size).rows2columnar(["languages", "text"])
dev_datapipe = dev_datapipe.map(batch_transform)
test_datapipe = test_datapipe.batch(batch_size).rows2columnar(["text"])
test_datapipe = test_datapipe.map(batch_transform_test)

train_dataloader = DataLoader(train_datapipe, batch_size=None)
dev_dataloader = DataLoader(dev_datapipe, batch_size=None)
test_dataloader = DataLoader(test_datapipe, batch_size=None)

print("First dev. batch:")
first_dev = next(iter(dev_dataloader))
print(first_dev)
# print("First test batch:")
# first_test = next(iter(test_dataloader))
# print(first_test)

# 3. Model definition
num_classes = len(labels_dict)
input_dim = 768

classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=input_dim)
model = XLMR_BASE_ENCODER.get_model(head=classifier_head)
model.to(DEVICE)

# 4. Train and save the model
import torchtext.functional as F
from torch.optim import AdamW

learning_rate = 5e-5
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

num_epochs = int(sys.argv[2])

for e in range(num_epochs):
    for batch in train_dataloader:
        input = F.to_tensor(batch["token_ids"], padding_value=padding_idx).to(DEVICE)
        target = torch.tensor(batch["target"]).to(DEVICE)
        train_step(input, target)
    loss, accuracy = evaluate()
    print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(e, loss, accuracy))


torch.save(model, "{}/{}.pth".format(model_base_path, language))

# 5. Predict on the test set and write the result
prediction = []
for batch in test_dataloader:
    input = F.to_tensor(batch["token_ids"], padding_value=padding_idx).to(DEVICE)
    output = model(input)
    _, z = torch.max(output.data, 1)
    prediction.extend(z.cpu().tolist())

print(prediction)

with open("{}/out/{}-open-VLP-run-3.txt".format(dsl, language), 'w') as f:
    for k in prediction:
        f.write(f"{labels_map[k]}\n")
