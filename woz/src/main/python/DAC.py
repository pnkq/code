from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

# 1. load the corpus
data_folder = '/home/phuonglh/code/woz/dat/woz/nlu/act'

# 2. what label do we want to predict?
label_type = 'act1'

column_name_map = {3: "text", 2: "act2", 1: "act1"}
corpus: Corpus =  CSVClassificationCorpus(data_folder, column_name_map, label_type=label_type, delimiter='\t')

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# # 4. initialize embeddings
embeddings = WordEmbeddings('glove')
