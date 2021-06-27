import pandas as pd
from torchtext.legacy import data
import torchtext
import random

class QuoraDuplicateQuestionsDataset():
    def __init__(self, root_directory="data", test_split = 15, transform=False, augment_probability=0.2):
        SEED = 1234
        
        # reading the dataset from the common csv files from each folder
        quora_dataframe = pd.read_csv(f"{root_directory}/quora_duplicate_questions.tsv", sep="	")
        
        # making source and target fields, with spacy as tokenizer
        self.Source = data.Field(tokenize = 'spacy', init_token = '<sos>', eos_token = '<eos>', lower = True)
        self.Target = data.Field(tokenize = 'spacy', init_token = '<sos>', eos_token = '<eos>', lower = True)
        fields = [('source', self.Source),('target', self.Target)]
        
        # cleaning the dataset
        # taking only duplicate questions from the dataset and making an example out of it based on fields
        quora_dataframe = quora_dataframe.query("is_duplicate==1")
        quora_dataframe = quora_dataframe.set_index(pd.Index(range(quora_dataframe.shape[0]))) # setting incremental indices
        example = [torchtext.legacy.data.Example.fromlist([quora_dataframe.question1[i],quora_dataframe.question2[i]], fields) for i in range(quora_dataframe.shape[0])] 
        
        # creating dataset out of the information available
        quora_dataset = torchtext.legacy.data.Dataset(example, fields)
        
        # splitting the dataset into train and validation (read test) datasets
        (self.train_dataset, self.validation_dataset) =  quora_dataset.split(split_ratio=[100-test_split, test_split], random_state = random.seed(SEED))
        self._build_vocab()
        
        
    def _build_vocab(self):
        # please do that only for your train dataset
        self.Source.build_vocab(self.train_dataset, min_freq = 2), self.Target.build_vocab(self.train_dataset, min_freq = 2)