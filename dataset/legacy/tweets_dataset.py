import pandas as pd
from utils.augmentations import processTweet
from torchtext.legacy import data
import torchtext
import random

class TweetsDataset():
    def __init__(self, root_directory="data", test_split = 15, transform=False, augment_probability=0.2):
        SEED = 1234
        tweets_dataframe = pd.read_csv(f"{root_directory}/tweets.csv")
        self.Tweet = data.Field(sequential = True, tokenize = 'spacy', batch_first =True, include_lengths=True)
        self.Label = data.LabelField(tokenize ='spacy', is_target=True, batch_first =True, sequential =False)
        fields = [('review', self.Tweet),('rating', self.Label)]
        # cleaning the dataset
        tweets_dataframe["tweets"] = tweets_dataframe["tweets"].apply(lambda tweet: processTweet(tweet))
        
        example = [torchtext.legacy.data.Example.fromlist([tweets_dataframe.tweets[i],tweets_dataframe.labels[i]], fields) for i in range(tweets_dataframe.shape[0])] 
        
        twitter_dataset = torchtext.legacy.data.Dataset(example, fields)
        (self.train_dataset, self.validation_dataset) =  twitter_dataset.split(split_ratio=[100-test_split, test_split], random_state = random.seed(SEED))
        self._build_vocab()
        
        
    def _build_vocab(self):
        # please do that only for your train dataset
        self.Tweet.build_vocab(self.train_dataset), self.Label.build_vocab(self.train_dataset)