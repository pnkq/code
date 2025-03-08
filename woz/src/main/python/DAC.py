from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from flair.embeddings import WordEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

# 1. load the corpus
columns = {0: 'token', 1: 'ner'}

data_folder = '/Users/phuonglh/code/woz/dat/woz/nlu/act'

# 2. what label do we want to predict?
label_type = 'topic'

corpus: Corpus =  ClassificationCorpus(data_folder, label_type=label_type)

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# # 4. initialize embeddings
# embeddings = WordEmbeddings('glove')
