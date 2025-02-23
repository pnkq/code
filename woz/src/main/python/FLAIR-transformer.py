from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
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
embeddings = TransformerWordEmbeddings(model='xlm-roberta-large', layers="-1", subtoken_pooling="first", fine_tune=True, use_context=True)

# 5. initialize sequence tagger
model = SequenceTagger(hidden_size=256, embeddings=embeddings, tag_dictionary=label_dict, tag_type=label_type, use_crf=False, use_rnn=False, reproject_embeddings=False)

# 6. initialize trainer
trainer = ModelTrainer(model, corpus)

# 7. start training
trainer.train('taggers/woz-xlmr-256', learning_rate=5e-6, mini_batch_size=32, max_epochs=100)

