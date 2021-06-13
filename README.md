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

## Inferencing
### Misclassified predictions
```
----
Tweet : RT @wardollarshome: Obama has approved more targeted assassinations than any modern US prez; READ & RT: http://t.co/bfC4gbBW

Cleaned tweet : rt AT USER obama has approved more targeted assassinations than any modern us prez read rt URL

Prediction: Negative	Actual: Positive


----
Tweet : one Chicago kid who says "Obama is my man" tells Jesse Watters that the gun violence in Chicago is like "World War 17"

Cleaned tweet : one chicago kid who says obama is my man tells jesse watters that the gun violence in chicago is like world war 17

Prediction: Positive	Actual: Negative


----
Tweet : #WhatsRomneyHiding? Obama's dignity and sense of humor? #p2 #tcot

Cleaned tweet : whatsromneyhiding obamas dignity and sense of humor p2 tcot

Prediction: Negative	Actual: Neutral


----
Tweet : Here's How Obama and the Democrats Will Win in 2012: Let's start by going back to the assorted polls questioning... http://t.co/zpg0TVm3

Cleaned tweet : heres how obama and the democrats will win in 2012 lets start by going back to the assorted polls questioning URL

Prediction: Negative	Actual: Positive


----
Tweet : RealClearPolitics - Obama's Organizational Advantage on Full ...: As a small but electorally significant state t... http://t.co/3Ax22aBB

Cleaned tweet : realclearpolitics obamas organizational advantage on full as a small but electorally significant state t URL

Prediction: Positive	Actual: Neutral


----
Tweet : RT @FrankConniff: Harvard graduate Romney slammed Obama for going to Harvard. Good to know that Mitt Romney hates people like Mitt Romney.

Cleaned tweet : rt AT USER harvard graduate romney slammed obama for going to harvard good to know that mitt romney hates people like mitt romney

Prediction: Negative	Actual: Positive


----
Tweet : RT @wilycyotee Pres. Obama's ongoing support of women is another reason I am so proud he is my President!  @edshow #Obama2012

Cleaned tweet : rt AT USER pres obamas ongoing support of women is another reason i am so proud he is my president AT USER obama2012

Prediction: Negative	Actual: Neutral


----
Tweet : RT @RandallHoven: Nixon was missing 18 minutes. Obama is missing 18 years.

Cleaned tweet : rt AT USER nixon was missing 18 minutes obama is missing 18 years

Prediction: Positive	Actual: Negative


----
Tweet : #WhatsRomneyHiding Obama's birth certificate.

Cleaned tweet : whatsromneyhiding obamas birth certificate

Prediction: Positive	Actual: Negative


----
Tweet : Thanks for the shout out in this recap of #JOBSAct signing! http://t.co/njCfZuSd via @MiamiHerald

Cleaned tweet : thanks for the shout out in this recap of jobsact signing URL via AT USER

Prediction: Positive	Actual: Negative


----
Tweet : #edshow  My President is standing up,speaking up,fighting back and I am so glad he has taken the gloves off. OBAMA 2012 @edshow

Cleaned tweet : edshow my president is standing up speaking up fighting back and i am so glad he has taken the gloves off obama 2012 AT USER

Prediction: Negative	Actual: Positive


----
Tweet : If Obama win 2012 Election wait til 2016 he will have full white hair! just like Bill clinton!

Cleaned tweet : if obama win 2012 election wait til 2016 he will have full white hair just like bill clinton

Prediction: Positive	Actual: Neutral


----
Tweet : RT @Un_Progressive: .@ResistTyranny @freegalt Easy. Obama is hiding everything. Always has.

Cleaned tweet : rt AT USER AT USER AT USER easy obama is hiding everything always has

Prediction: Negative	Actual: Positive


----
Tweet : Even CBS won't buy bogus WH explanation of Obama Supreme Court comments - at http://t.co/rkNdEmIy #withnewt #tpn #tcot #tlot #tpp #sgp

Cleaned tweet : even cbs wont buy bogus wh explanation of obama supreme court comments at URL withnewt tpn tcot tlot tpp sgp

Prediction: Negative	Actual: Positive


----
Tweet : #whatsObamahiding Kryptonite for the Grand Old Party! Obama 2012

Cleaned tweet : whatsobamahiding kryptonite for the grand old party obama 2012

Prediction: Negative	Actual: Positive


----
Tweet : RT @BunkerBlast: RT @teacherspets Obama's Budget: 'Interest Payments Will Exceed Defense Budget' in 2019  http://t.co/uddCXCjt

Cleaned tweet : rt AT USER rt AT USER obamas budget interest payments will exceed defense budget in 2019 URL

Prediction: Negative	Actual: Neutral


----
Tweet : @edshow The Right has been crying about unelected judges legislation from  bench forever all of a sudden Obama says same thing he is a Thug

Cleaned tweet : AT USER the right has been crying about unelected judges legislation from bench forever all of a sudden obama says same thing he is a thug

Prediction: Negative	Actual: Positive
```
### Correct predictions
```
----
Tweet : Obama has called the GOP budget social Darwinism. Nice try, but they believe in social creationism.

Cleaned tweet : obama has called the gop budget social darwinism nice try but they believe in social creationism

Prediction: Positive	Actual: Positive


----
Tweet : In his teen years, Obama has been known to use marijuana and cocaine.

Cleaned tweet : in his teen years obama has been known to use marijuana and cocaine

Prediction: Negative	Actual: Negative


----
Tweet : IPA Congratulates President Barack Obama for Leadership Regarding JOBS Act: WASHINGTON, Apr 05, 2012 (BUSINESS W... http://t.co/8le3DC8E

Cleaned tweet : ipa congratulates president barack obama for leadership regarding jobs act washington apr 05 2012 business w URL

Prediction: Negative	Actual: Negative


----
Tweet : RT @Professor_Why: #WhatsRomneyHiding - his connection to supporters of Critical Race Theory.... Oh wait, that was Obama, not Romney...

Cleaned tweet : rt AT USER whatsromneyhiding his connection to supporters of critical race theory oh wait that was obama not romney

Prediction: Negative	Actual: Negative


----
Tweet : Video shows federal officials joking about cost of lavish conference http://t.co/2i4SmoPM #obama #crime #p2 #news #tcot #teaparty

Cleaned tweet : video shows federal officials joking about cost of lavish conference URL obama crime p2 news tcot teaparty

Prediction: Negative	Actual: Negative


----
Tweet : RT @ohgirlphrase: American kid "You're from the UK? Ohhh cool, So do you have tea with the Queen?". British kid: "Do you like, go to Mcdonalds with Obama?

Cleaned tweet : rt AT USER american kid youre from the uk ohhh cool so do you have tea with the queen british kid do you like go to mcdonalds with obama

Prediction: Negative	Actual: Negative


----
Tweet : A valid explanation for why Obama won't let women on the golf course.   #WhatsRomneyHiding

Cleaned tweet : a valid explanation for why obama wont let women on the golf course whatsromneyhiding

Prediction: Positive	Actual: Positive


----
Tweet : President Obama &lt; Lindsay Lohan RUMORS beginning cross shape lights on ST &lt; 1987 Analyst64 DC bicycle courier &lt; Video changes to scramble.

Cleaned tweet : president obama lt lindsay lohan rumors beginning cross shape lights on st lt 1987 analyst64 dc bicycle courier lt video changes to scramble

Prediction: Negative	Actual: Negative


----
Tweet : Obama's Gender Advantage Extends to the States - 2012 Decoded: New detail on recent swing state polling further ... http://t.co/8iSanDGS

Cleaned tweet : obamas gender advantage extends to the states 2012 decoded new detail on recent swing state polling further URL

Prediction: Positive	Actual: Positive


----
Tweet : RT @GregWHoward: Obama inherited (if you want to use his lame wording) $10.6 trillion in debt and turned it into $15.2 trillion. #tcot #p2 #ocra #teaparty

Cleaned tweet : rt AT USER obama inherited if you want to use his lame wording 10 6 trillion in debt and turned it into 15 2 trillion tcot p2 ocra teaparty

Prediction: Negative	Actual: Negative


----
Tweet : RT @ohgirlphrase: American kid "You're from the UK? Ohhh cool, So do you have tea with the Queen?". British kid: "Do you like, go to Mcdonalds with Obama?

Cleaned tweet : rt AT USER american kid youre from the uk ohhh cool so do you have tea with the queen british kid do you like go to mcdonalds with obama

Prediction: Negative	Actual: Negative


----
Tweet : of course..I blame HAARP, sinners, aliens, Bush and Obama for all this.  :P....&gt;&gt; http://t.co/7eq8nebt

Cleaned tweet : of course i blame haarp sinners aliens bush and obama for all this p gt gt URL

Prediction: Negative	Actual: Negative


----
Tweet : Pravda Endorses and Supports Obama   http://t.co/G68JEmkz

Cleaned tweet : pravda endorses and supports obama URL

Prediction: Negative	Actual: Negative


----
Tweet : RT @mlake9: Love how the #Obama campaign's #WhatsRomneyHiding back fired, now #WhatsObamaHiding is trending. Seeing a lot of birth certificate tweets.

Cleaned tweet : rt AT USER love how the obama campaigns whatsromneyhiding back fired now whatsobamahiding is trending seeing a lot of birth certificate tweets

Prediction: Positive	Actual: Positive


----
Tweet : @katie_larson It was Obama's hashtag--conservatives turned it into a game (again) :) It's funny ;)

Cleaned tweet : AT USER it was obamas hashtag conservatives turned it into a game again its funny

Prediction: Negative	Actual: Negative


----
Tweet : RT @ohgirlphrase: American kid "You're from the UK? Ohhh cool, So do you have tea with the Queen?". British kid: "Do you like, go to Mcdonalds with Obama?

Cleaned tweet : rt AT USER american kid youre from the uk ohhh cool so do you have tea with the queen british kid do you like go to mcdonalds with obama

Prediction: Negative	Actual: Negative


----
Tweet : RT @mlake9: Love how the #Obama campaign's #WhatsRomneyHiding back fired, now #WhatsObamaHiding is trending. Seeing a lot of birth certificate tweets.

Cleaned tweet : rt AT USER love how the obama campaigns whatsromneyhiding back fired now whatsobamahiding is trending seeing a lot of birth certificate tweets

Prediction: Positive	Actual: Positive


----
Tweet : @edshow the direspect of President #Obama is based on racism. They do not want a Black PRESIDENT. #edshow

Cleaned tweet : AT USER the direspect of president obama is based on racism they do not want a black president edshow

Prediction: Negative	Actual: Negative


----
Tweet : RT @Professor_Why: #WhatsRomneyHiding - his plan to push bad loans through for Green energy just to see 'em go bankrupt... Oops, that's Obama again!

Cleaned tweet : rt AT USER whatsromneyhiding his plan to push bad loans through for green energy just to see em go bankrupt oops thats obama again

Prediction: Negative	Actual: Negative


----
Tweet : RT @haymakers: When RWers whine "What's Obama doing about gas prices!?" Tell them, "Nothing. That's how the free-market works. What are you? A socialist?"

Cleaned tweet : rt AT USER when rwers whine whats obama doing about gas prices tell them nothing thats how the free market works what are you a socialist

Prediction: Negative	Actual: Negative


----
Tweet : RT @0ryuge: #WhatsRomneyHiding The videotape of Barack Obama sucking up to radical Rashid Khalidi. Oh wait, that's the LATimes #VetThePress #VetThePres

Cleaned tweet : rt AT USER whatsromneyhiding the videotape of barack obama sucking up to radical rashid khalidi oh wait thats the latimes vetthepress vetthepres

Prediction: Negative	Actual: Negative


----
Tweet : RT @EMcLean1982: #WhatsRomneyHiding Obama's Mom Jeans

Cleaned tweet : rt AT USER whatsromneyhiding obamas mom jeans

Prediction: Negative	Actual: Negative


----
Tweet : RT @Libertarian_76: I can't wait until Romney goes after Obama on NDAA. Oh wait.

Cleaned tweet : rt AT USER i cant wait until romney goes after obama on ndaa oh wait

Prediction: Negative	Actual: Negative


----
Tweet : #KimKardashiansNextBoyFriend Obama lol Michelle ain't goin 4 dat

Cleaned tweet : kimkardashiansnextboyfriend obama lol michelle aint goin 4 dat

Prediction: Negative	Actual: Negative


----
Tweet : Obama And The Democrats Shocked By Constitutional Restraint http://t.co/RhSzoEU7 via @IBDinvestors

Cleaned tweet : obama and the democrats shocked by constitutional restraint URL via AT USER

Prediction: Positive	Actual: Positive
```

## "The Actual Assignment"
```
Input sentence : 'Obama is a very very bad person'

Predicted : 'Negative'

Actual : 'Negative'


Encoder 0
tensor([[-0.3326, -0.0291,  0.0139,  0.1516,  0.0039, -0.0206, -0.4432,  0.0555,
          0.4517,  0.5568, -0.6531,  0.2773,  0.0017, -0.2519,  0.0504,  0.0926,
          0.5178, -0.6054,  0.1508, -0.1905]], grad_fn=<MulBackward0>)
Encoder 1
tensor([[-0.4290, -0.5466,  0.2733, -0.0392,  0.2070,  0.3221,  0.0357, -0.0811,
          0.5935,  0.0762, -0.2338,  0.0524,  0.0108, -0.0077,  0.3277, -0.1900,
          0.0007, -0.4905, -0.0241,  0.1239]], grad_fn=<MulBackward0>)
Encoder 2
tensor([[-0.0246,  0.1054, -0.0060,  0.0719,  0.0039, -0.1117, -0.0065,  0.6118,
          0.0457, -0.0303, -0.1856, -0.1946, -0.2983,  0.4614,  0.0203, -0.0078,
         -0.3902, -0.5649,  0.0283,  0.6676]], grad_fn=<MulBackward0>)
Encoder 3
tensor([[-3.2853e-01, -4.2382e-03, -2.5963e-03,  3.7073e-02,  5.3789e-01,
         -3.1684e-02, -2.0090e-03,  1.6923e-01,  2.6905e-01, -5.0071e-01,
          9.9603e-02, -3.4340e-01, -9.9730e-02,  6.4148e-02,  1.3340e-01,
          2.1384e-01,  4.2437e-05,  3.4066e-02,  4.1255e-01, -2.0025e-01]],
       grad_fn=<MulBackward0>)
Encoder 4
tensor([[-0.3761, -0.0054, -0.0029, -0.0196,  0.5653, -0.0081, -0.0037,  0.1267,
          0.3298, -0.4718,  0.1118, -0.3542, -0.1830,  0.0503,  0.2147,  0.2975,
          0.0038,  0.0155,  0.2648, -0.2956]], grad_fn=<MulBackward0>)
Encoder 5
tensor([[-8.6374e-02,  3.9941e-01,  3.4409e-02,  3.1015e-01,  4.5534e-01,
          2.6947e-01, -1.8570e-02,  1.7552e-03,  7.9063e-01, -2.2716e-01,
         -4.5663e-04,  2.3911e-01, -2.0369e-01,  1.8628e-01,  3.9019e-01,
         -9.5272e-03, -2.0034e-02,  1.0342e-02,  8.4490e-02,  8.3634e-02]],
       grad_fn=<MulBackward0>)
Encoder 6
tensor([[-2.0857e-01,  2.5747e-02,  3.8452e-02,  2.2480e-01,  6.5218e-01,
         -2.8299e-02, -6.3507e-02,  8.2817e-02,  6.4707e-01,  2.4353e-02,
         -1.9128e-01,  7.0194e-01, -4.7980e-01, -5.4766e-01,  1.2887e-01,
         -2.1756e-01,  1.0344e-01,  1.1598e-01,  5.6316e-04,  5.3311e-01]],
       grad_fn=<MulBackward0>)
Decoder 0
tensor([[-0.2222,  0.0951, -0.1535,  0.2013,  0.3357,  0.0079, -0.1670, -0.0396,
          0.3630,  0.0898, -0.0132,  0.1777, -0.3049, -0.1276,  0.1913,  0.0709,
          0.1360,  0.1365,  0.0934,  0.1821]], grad_fn=<MulBackward0>)
Decoder 1
tensor([[-0.3049,  0.1173, -0.2649, -0.0759,  0.3724,  0.0600, -0.2648, -0.1181,
          0.3479,  0.0767,  0.0879, -0.0585, -0.3104,  0.0211,  0.0867,  0.2278,
          0.2086,  0.2072,  0.1786,  0.1816]], grad_fn=<MulBackward0>)
Decoder 2
tensor([[-0.3896,  0.1719, -0.3543, -0.2880,  0.3990,  0.0805, -0.3613, -0.1916,
          0.3664,  0.0414,  0.1558, -0.2083, -0.3488,  0.1638,  0.0171,  0.3786,
          0.3046,  0.2986,  0.2935,  0.2466]], grad_fn=<MulBackward0>)
Decoder 3
tensor([[-0.4655,  0.2433, -0.4300, -0.4355,  0.4372,  0.0957, -0.4568, -0.2615,
          0.4106, -0.0035,  0.2080, -0.3092, -0.4028,  0.2802, -0.0401,  0.5170,
          0.3972,  0.3911,  0.4014,  0.3197]], grad_fn=<MulBackward0>)
Decoder 4
tensor([[-0.5286,  0.3151, -0.4964, -0.5320,  0.4784,  0.1096, -0.5439, -0.3265,
          0.4617, -0.0476,  0.2506, -0.3843, -0.4555,  0.3665, -0.0920,  0.6309,
          0.4751,  0.4733,  0.4882,  0.3827]], grad_fn=<MulBackward0>)
```
## Conclusion
Our model is too much of an overkill given the data we have at our end. Regularization techniques like dropout, augmentations are a must if we want to have a nice validation accuracy.