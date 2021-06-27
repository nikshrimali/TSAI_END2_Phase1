import pandas as pd
from torchtext.legacy import data
import torchtext
import random

class CmuQAndADataset():
    def __init__(self, root_directory="data/Question_Answer_Dataset_v1.2", filename="question_answer_pairs.txt", test_split = 15, transform=False, augment_probability=0.2):
        SEED = 1234
        # reading the dataset from the common csv files from each folder
        cmu_s08 = pd.read_csv(f'{root_directory}/S08/{filename}', sep='\t', encoding = "ISO-8859-1")
        cmu_s09 = pd.read_csv(f'{root_directory}/S09/{filename}', sep='\t', encoding = "ISO-8859-1")
        cmu_s10 = pd.read_csv(f'{root_directory}/S10/{filename}', sep='\t', encoding = "ISO-8859-1")
        
        # combining the above csv files read 
        complete_cmu_dataframe = pd.concat([cmu_s08, cmu_s09, cmu_s10])
        
        # making source and target fields, with spacy as tokenizer
        self.Source = data.Field(tokenize = 'spacy', init_token = '<sos>', eos_token = '<eos>', lower = True)
        self.Target = data.Field(tokenize = 'spacy', init_token = '<sos>', eos_token = '<eos>', lower = True)
        fields = [('source', self.Source),('target', self.Target)]
        
        # cleaning the dataset
        complete_cmu_dataframe = complete_cmu_dataframe.dropna(axis=0) # drop NaN values
        complete_cmu_dataframe = complete_cmu_dataframe.set_index(pd.Index(range(complete_cmu_dataframe.shape[0]))) # setting incremental indices

        # taking only question and answer from the dataset and making an example out of it based on fields
        example = [torchtext.legacy.data.Example.fromlist([complete_cmu_dataframe.Question[i],complete_cmu_dataframe.Answer[i]], fields) for i in range(complete_cmu_dataframe.shape[0])] 
        
        # creating dataset out of the information available
        cmu_wiki_dataset = torchtext.legacy.data.Dataset(example, fields)
        
        # splitting the dataset into train and validation (read test) datasets
        (self.train_dataset, self.validation_dataset) =  cmu_wiki_dataset.split(split_ratio=[100-test_split, test_split], random_state = random.seed(SEED))
        
        # making sure vocabulary is built on train dataset 
        self._build_vocab()
        
        
    def _build_vocab(self):
        # please do that only for your train dataset
        self.Source.build_vocab(self.train_dataset, min_freq = 2), self.Target.build_vocab(self.train_dataset, min_freq = 2)