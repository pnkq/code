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
model_name="bert-large-cased"
#model_name="xlm-roberta-large"
#model_name="roberta-large"
#model_name="microsoft/deberta-v3-base"
embeddings = TransformerWordEmbeddings(model=model_name, layers="-1", subtoken_pooling="first", fine_tune=False, use_context=False)

# 5. initialize sequence tagger
hidden_size=512
model = SequenceTagger(hidden_size=hidden_size, embeddings=embeddings, tag_dictionary=label_dict, tag_type=label_type, use_crf=False, use_rnn=True, reproject_embeddings=False)

# 6. initialize trainer
trainer = ModelTrainer(model, corpus)

# 7. start fine-tuning
trainer.fine_tune(f'taggers-featurize/{model_name}-{hidden_size}', learning_rate=1e-3, mini_batch_size=16, max_epochs=100)

