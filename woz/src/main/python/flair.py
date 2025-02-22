from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# 1. load the corpus
columns = {0: 'token', 1: 'ner'}

data_folder = '/home/phuonglh/code/woz/dat/woz/nlu/'
corpus: Corpus = ColumnCorpus(data_folder, columns, train_file='train.txt', dev_file='dev.txt', test_file='test.txt')

# 2. what label do we want to predict?
label_type = 'ner'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# 4. initialize embeddings
embeddings = WordEmbeddings('glove')

# 5. initialize sequence tagger
model = SequenceTagger(hidden_size=256, embeddings=embeddings, tag_dictionary=label_dict, tag_type=label_type)

# 6. initialize trainer
trainer = ModelTrainer(model, corpus)

# 7. start training
trainer.train('taggers/woz', learning_rate=0.1, mini_batch_size=64, max_epochs=100)

