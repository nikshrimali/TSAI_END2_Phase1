# Assignment 6 - Encoder Decoder 
> Submitted by Nikhil Shrimali
## Target

* Take the last code (+tweet dataset) and convert that in such a war that:
    * encoder: an RNN/LSTM layer takes the words in a sentence one by one and finally converts them into a single vector. **VERY IMPORTANT TO MAKE THIS SINGLE VECTOR**
    * this single vector is then sent to another RNN/LSTM that also takes the last prediction as its second input. Then we take the final vector from this Cell
    * and send this final vector to a Linear Layer and make the final prediction. 
    * This is how it will look:
        * embedding
        * word from a sentence +last hidden vector -> encoder -> single vector
        * single vector + last hidden vector -> decoder -> single vector
        * single vector -> FC layer -> Prediction
* Your code will be checked for plagiarism, and if we find that you have copied from the internet, then -100%. 
* The code needs to look as simple as possible, the focus is on making encoder/decoder classes and how to link objects together
* Getting good accuracy is NOT the target, but must achieve at least 45% or more
* Once the model is trained, take one sentence, "print the outputs" of the encoder for each step and "print the outputs" for each step of the decoder. ← THIS IS THE ACTUAL ASSIGNMENT

## Submission
I have trained model, and observations, details can be found below. You can see detailed and step by step of explainination of every step in [this Jupyter Notebook](encoder-decoder.ipynb) 

---

## Dataset
This assignment is all about working on twitter dataset (it's sorted and easy !)

### Broken dreams
I successfully failed to make my own dataset without having to use the "legacy" torchtext. You can have a look into it by checking the [custom_tweets_dataset.py](dataset\custom_tweets_dataset.py) file

### Grief acceptance
Once a great man said "You either try being a hero, or fail long enough to see yourself become the villan". 

```python
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
```

Please refer the [legacy tweets_dataset](dataset\legacy\tweets_dataset.py) file for more info

### Dataset parameters
The Dataset consists of `1341 (Cleaned) Tweets` which are labelled [Negative, Positive, Neutral]

`Negative: 931, Positive: 352, Neutral: 81`

---

## Loss Function
Since it is indeed a classification problem, I'm using the old tried and tested `CrossEntropyLoss`

---

## Model
This is the highlight of this assignment ! I've used an encoder-decoder architecture for this assignment

Both of the encoder and the decoder used in this model are lstm. You can however change them to RNNs or GRUs according to your use case
### Defining encoder and decoder
```python
class Lstm(nn.Module):
    
    # Define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim = 16, staggered_input = False):
        
        super().__init__()          
        
        # Embedding layer
        if staggered_input:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM layer
        self.lstm_cell = nn.LSTMCell(input_size = embedding_dim, hidden_size=hidden_dim, bias=False)
        # Dense layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.staggered_input = staggered_input
        # staggered_input is basically identifier whether lstm class is of encoder or decoder. 
        # You're probably wondering why not just say so instead of naming it all complex-y. To that I'd say MERI MARZI
        
    def forward(self, source_text, num_steps, hidden, cell):
        for batch in range(source_text.shape[0]):
            output = []
            input = source_text[batch]
            if self.staggered_input:
                input = self.embedding(input)
                iter_steps = num_steps[batch]
            else:
                iter_steps = num_steps
            for index in range(iter_steps):
                if self.staggered_input:
                    encoder_input = input[index].unsqueeze(dim=0)
                    hidden, cell = self.lstm_cell(encoder_input, (hidden, cell))
                    output = hidden  
                else:
                    hidden, cell = self.lstm_cell(input.unsqueeze(dim=0), (hidden, cell))
                    # output = output.append(hidden)
                    output = hidden
            
        dense_outputs = self.fc(torch.stack(output, dim=0) if isinstance(output, list) else output) 
        # sending the last hidden output (which has seen everything) if encoder else sending decoder output
            
        return dense_outputs, (hidden, cell)
```

Next comes the part where we stitch the model all together. This involves
* Making embeddings out of input
* Sending the input embeddings to encoder (read LSTM here)
* Taking the output of encoder (last hidden state) and sending it to decoder (also, LSTM here).

### Defining the Encoder-Decoder model
```python
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, num_classes) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.linear_layer = nn.Linear(self.decoder.fc.out_features, num_classes)
        
    def forward(self, input, input_length, decoder_steps=5):
        hidden = cell = torch.zeros(1, self.encoder.lstm_cell.hidden_size, device= input.device)
        
        encoder_output, (hidden, cell) = self.encoder(input, input_length, hidden, cell)
        
        decoder_output, (hidden, cell) = self.decoder(encoder_output, decoder_steps, hidden, cell) 
        # decoder_steps signifies number of lstm cells used in the decoder. Default value is 5
        
        return self.linear_layer(decoder_output)
```

### Constructing encoder and decoder with parameters
```python
encoder = Lstm(vocab_size = size_of_vocab, embedding_dim= embedding_dim, hidden_dim = num_hidden_nodes, staggered_input = True)

decoder = Lstm(vocab_size = None, embedding_dim= encoder.fc.out_features, hidden_dim = num_hidden_nodes, output_dim = num_hidden_nodes, staggered_input = False)
```

### Constructing Encoder-Decoder model
```python
model = EncoderDecoder(encoder=encoder, decoder=decoder, num_classes=3, decoder_steps=5) 
# you can have custom number of classes here. decoder_steps signifies number of lstm cells used in the decoder
```
---

## Hyper-parameter tuning and results

| Hidden nodes (Encoder) | Hidden nodes (Decoder)  |  Epochs | Train Accuracy | Test Accuracy | Comments
| :---:   | :-: | :-: |  :-: |  :-: | :-: |
| 100 | 100 | 10 | 99.31% | 82.93% | Highly overfitting, need to regularize model
| 50 | 50 | 10 | 99.40% | 82.44% | Highly overfitting, need to regularize model
| 10 | 10 | 10 | 95.69% | 77.56% | Slightly less overfitting, but our validation accuracy also suffered

## Inferencing


## Training logs
```
    Train Loss: 0.775 | Train Acc: 67.73%
	 Val. Loss: 0.618 |  Val. Acc: 75.61% 

	Train Loss: 0.497 | Train Acc: 80.76%
	 Val. Loss: 0.525 |  Val. Acc: 83.90% 

	Train Loss: 0.204 | Train Acc: 92.84%
	 Val. Loss: 0.700 |  Val. Acc: 79.51% 

	Train Loss: 0.091 | Train Acc: 96.64%
	 Val. Loss: 0.817 |  Val. Acc: 79.02% 

	Train Loss: 0.037 | Train Acc: 98.71%
	 Val. Loss: 1.153 |  Val. Acc: 80.98% 

	Train Loss: 0.052 | Train Acc: 98.79%
	 Val. Loss: 0.960 |  Val. Acc: 79.02% 

	Train Loss: 0.032 | Train Acc: 98.62%
	 Val. Loss: 1.079 |  Val. Acc: 79.02% 

	Train Loss: 0.035 | Train Acc: 98.96%
	 Val. Loss: 1.166 |  Val. Acc: 80.98% 

	Train Loss: 0.023 | Train Acc: 99.31%
	 Val. Loss: 1.169 |  Val. Acc: 80.49% 

	Train Loss: 0.031 | Train Acc: 98.88%
	 Val. Loss: 0.895 |  Val. Acc: 82.93% 
```


## Conclusion
Our model is too much of an overkill given the data we have at our end. Regularization techniques like dropout, augmentations are a must if we want to have a nice validation accuracy.