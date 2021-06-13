from torch.utils.data import Dataset
import pandas as pd
from utils.augmentations import apply_augmentations, processTweet
from torchtext.legacy import data

class TweetsDataset(Dataset):
    def __init__(self, root_directory="data", train=False, test_split = 0.2, transform=False, augment_probability=0.2):
        
        self.tweets_dataframe = pd.read_csv(f"{root_directory}/tweets.csv")
        self.Tweet = data.Field(sequential = True, tokenize = 'spacy', batch_first =True, include_lengths=True)
        self.Label = data.LabelField(tokenize ='spacy', is_target=True, batch_first =True, sequential =False)
        self.fields = [('review', self.Tweet),('rating', self.Label)]
        # cleaning the dataset
        self.tweets_dataframe["tweets"] = self.tweets_dataframe["tweets"].apply(lambda tweet: processTweet(tweet))
        if train:
            self.train= True
            self.dataframe = self.tweets_dataframe.sample(frac = 1-test_split)
        else:
            self.train=False
            self.dataframe = self.tweets_dataframe.sample(frac = test_split)
            print("Sucessfully loaded twitter dataset for testing")
        
    def __len__(self):
        return self.dataframe.__len__()
        
    def __getitem__(self, index: int):
        return data.Example.fromlist([self.dataframe["tweets"].iloc[index],self.dataframe["labels"].iloc[index]], self.fields)
        
    def build_vocab(self):
        # please do that only for your train dataset
        return self.Tweet.build_vocab(self), self.Label.build_vocab(self)