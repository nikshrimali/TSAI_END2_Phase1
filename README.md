# Assignment 7 - Seq2Seq
> Submitted by Nikhil Shrimali

## Target
* Assignment 1 (500 points):
    * Submit the Assignment 5 as Assignment 1. To be clear,
        * ONLY use datasetSentences.txt. (no augmentation required)
        * Your dataset must have around 12k examples.
        * Split Dataset into 70/30 Train and Test (no validation)
        * Convert floating-point labels into 5 classes (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0) 
        * Upload to github and proceed to answer these questions asked in the S7 - Assignment Solutions, where these questions are asked:
            * Share the link to your github repo (100 pts for code quality/file structure/model accuracy)
            * Share the link to your readme file (200 points for proper readme file)
            * Copy-paste the code related to your dataset preparation (100 pts)
            * Share your training log text (you MUST have been testing for test accuracy after every epoch) (200 pts)
            * Share the prediction on 10 samples picked from the test dataset. (100 pts)
* Assignment 2 (300 points):
    * Train model we wrote in the class on the following two datasets taken from this [link](https://kili-technology.com/blog/chatbot-training-datasets/): 
        * [CMU Dataset](http://www.cs.cmu.edu/~ark/QA-data/)
        * [Quora Question Pairs Dataset](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)
    * Once done, please upload the file to github and proceed to answer these questions in the S7 - Assignment Solutions, where these questions are asked:
        * Share the link to your github repo (100 pts for code quality/file structure/model accuracy) (100 pts)
        * Share the link to your readme file (100 points for proper readme file), this file can be the second part of your Part 1 Readme (basically you can have only 1 Readme, describing both assignments if you want) (100 pts)
        * Copy-paste the code related to your dataset preparation for both datasets.  (100 pts)
## Submission
I have trained model, and observations, details can be found below. You can see detailed and step by step of explainination of everything in the below Jupyter Notebooks
* Training on [CMU Q&A DATASET](CMUQ&A.ipynb)
* Training on [Quora Duplicate Questions Dataset](Quora_Duplicate_Question.ipynb)

## CMU Q&A Dataset from _Wikipedia_

This corpus includes Wikipedia articles, factual questions manually generated from them, and answers to these manually generated questions for use in academic research.

### A brief look at the dataset 

| ArticleTitle | Question  |  Answer | DifficultyFromQuestioner | DifficultyFromAnswerer | ArticleFile
| :---:   | :-: | :-: |  :-: |  :-: | :-: |
Uruguay | What does a citizen use to propose changes to the Constitution? | Referendum |	medium |	medium |	data/set2/a9
Uruguay | What does a citizen use to propose changes to the Constitution?	 | Plebiscite |	medium | medium | data/set2/a9
Uruguay |	Where is Uruguay's oldest church?	San Carlos, Maldonado |	medium |	medium |	data/set2/a9

### Preparing the dataset
```python
class CmuQAndADataset():
    """
    It ain't much, but it's honest work
    """
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
```
You can find this class under the [dataset](dataset) directory

---

## Quora Question and Answer dataset
A set of Quora questions to determine whether pairs of question texts actually correspond to semantically equivalent queries. More than 400,000 lines of potential questions duplicate question pairs.

> Since our job is to generate a english response to a particular input, we are actually looking out for duplicate question pairs in this dataset, and making sure we get a lesser perplexity as we train our model

		
### A brief look at the dataset 
					

| id | qid1  |  qid2 | question1 | question2 | is_duplicate
| :---:   | :-: | :-: |  :-: |  :-: | :-: |
100 |	201 |	202 |	Will there really be any war between India and Pakistan over the Uri attack? What will be its effects? |	Will there be a nuclear war between India and Pakistan? |	1
101 |	203 |	204 |	Did Ronald Reagan have a mannerism in his speech? |	How did Ronald Reagan react to 9/11? |	0
102 |	205 |	206 |	What were the war strategies of the Union and the Confederates during the Civil War? |	How could the Confederates have possibly defeated Union forces at Gettysburg during the American Civil War? |	0

### Preparing the dataset
```python
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

```
You can find this class under the [dataset](dataset) directory

## Loss function
I've used `CrossEntropyLoss` for this problem as my approach for this assignment was to create a vocabulary out of the target duplicate question or answer (depending on the dataset chosen), and compare the output from the model. I'm still not sure if this was the right way to go, but this is where I ended up with.

## Model
I spent very less effort creating/optimising the model given. Get ready to see what you already saw - _Seq2Seq Model_ !
```
Seq2Seq(
  (encoder): Encoder(
    (embedding): Embedding(15834, 256)
    (rnn): LSTM(256, 512, num_layers=2, dropout=0.5)
    (dropout): Dropout(p=0.5, inplace=False)
  )
  (decoder): Decoder(
    (embedding): Embedding(15739, 256)
    (rnn): LSTM(256, 512, num_layers=2, dropout=0.5)
    (fc_out): Linear(in_features=512, out_features=15739, bias=True)
    (dropout): Dropout(p=0.5, inplace=False)
  )
)
```
The model had `23,513,211` trainable parameters

---

## Training over CMU Wikipedia Question and Answer dataset
### Logs
```
Epoch: 01 | Time: 0m 9s
	Train Loss: 4.868 | Train PPL: 130.021
	 Val. Loss: 3.910 |  Val. PPL:  49.892
Epoch: 02 | Time: 0m 6s
	Train Loss: 4.457 | Train PPL:  86.245
	 Val. Loss: 3.838 |  Val. PPL:  46.439
Epoch: 03 | Time: 0m 6s
	Train Loss: 4.369 | Train PPL:  78.981
	 Val. Loss: 3.953 |  Val. PPL:  52.090
Epoch: 04 | Time: 0m 6s
	Train Loss: 4.294 | Train PPL:  73.265
	 Val. Loss: 3.894 |  Val. PPL:  49.113
Epoch: 05 | Time: 0m 6s
	Train Loss: 4.205 | Train PPL:  67.026
	 Val. Loss: 3.894 |  Val. PPL:  49.109
Epoch: 06 | Time: 0m 6s
	Train Loss: 4.173 | Train PPL:  64.933
	 Val. Loss: 3.902 |  Val. PPL:  49.483
Epoch: 07 | Time: 0m 6s
	Train Loss: 4.133 | Train PPL:  62.364
	 Val. Loss: 3.862 |  Val. PPL:  47.566
Epoch: 08 | Time: 0m 6s
	Train Loss: 4.101 | Train PPL:  60.400
	 Val. Loss: 3.882 |  Val. PPL:  48.506
Epoch: 09 | Time: 0m 6s
	Train Loss: 4.057 | Train PPL:  57.824
	 Val. Loss: 3.919 |  Val. PPL:  50.361
Epoch: 10 | Time: 0m 6s
	Train Loss: 3.982 | Train PPL:  53.619
	 Val. Loss: 3.775 |  Val. PPL:  43.609
```
### Observations
* I trained it on my local machine, with the batch size of `32`
* As you can notice, training over this dataset took very less amount of time (average of 6 seconds), that too being trained on my local machine (Nvidia GTX1650 - 4GB Graphic card)
* While the Train  perplexity started at a  very high number, the validation one started from a small amount, but as training progressed, training perplexity decreased at a faster pace than that of validation, but both endeded up being in similar range. So, might not be overfitting. I tried training to higher epochs, but there was a little change in the overall trend

---

## Training over Quora Duplicate Question dataset
### Logs

```
Epoch: 01 | Time: 6m 8s
	Train Loss: 5.166 | Train PPL: 175.293
	 Val. Loss: 5.167 |  Val. PPL: 175.431
Epoch: 02 | Time: 6m 7s
	Train Loss: 4.377 | Train PPL:  79.596
	 Val. Loss: 4.907 |  Val. PPL: 135.269
Epoch: 03 | Time: 6m 6s
	Train Loss: 3.858 | Train PPL:  47.373
	 Val. Loss: 4.519 |  Val. PPL:  91.721
Epoch: 04 | Time: 6m 8s
	Train Loss: 3.445 | Train PPL:  31.343
	 Val. Loss: 4.294 |  Val. PPL:  73.271
Epoch: 05 | Time: 6m 8s
	Train Loss: 3.170 | Train PPL:  23.809
	 Val. Loss: 4.071 |  Val. PPL:  58.599
Epoch: 06 | Time: 6m 7s
	Train Loss: 2.960 | Train PPL:  19.292
	 Val. Loss: 4.072 |  Val. PPL:  58.669
Epoch: 07 | Time: 6m 5s
	Train Loss: 2.774 | Train PPL:  16.028
	 Val. Loss: 3.985 |  Val. PPL:  53.773
Epoch: 08 | Time: 6m 10s
	Train Loss: 2.635 | Train PPL:  13.943
	 Val. Loss: 3.948 |  Val. PPL:  51.810
Epoch: 09 | Time: 6m 7s
	Train Loss: 2.512 | Train PPL:  12.331
	 Val. Loss: 3.889 |  Val. PPL:  48.864
Epoch: 10 | Time: 6m 7s
	Train Loss: 2.419 | Train PPL:  11.231
	 Val. Loss: 3.885 |  Val. PPL:  48.643
```
### Observations
* I trained it on google colab, with the batch size of `256`
* As you can notice, training over this dataset took fairly large amount of time (average of 6 mins 7 seconds), that too on Google Colab (Tesla T4 - 16GB graphic card) (My machine did it in 16 minutes)
* Train and the validation perplexity started at a  very high number, but as training progressed, training perplexity decreased at a faster pace than that of validation. This makes me believe the model was overfitting, and required augmentations (I was lazy enough not to add it)

## Inferencing
I failed at inferencing the output after the model is being trained 😭. This is what I tried
```python
def generate_answer(question):
    #input is the question for which you need to generate answer (or duplicated question for quora dataset)
    # tokenize the question 
    tokenized = [tok.text for tok in nlp.tokenizer(question)]
    print(tokenized)
    # convert to integer sequence using predefined tokenizer dictionary
    indexed = [tokenizer[t] for t in tokenized]
    # convert to tensor          
    tensor = torch.LongTensor(indexed).to("cpu")   
    # reshape in form of batch, no. of words           
    tensor = tensor.unsqueeze(1)
    # Get the model prediction, which expects the prediction with same number of words as input                
    prediction = model(tensor, tensor).squeeze().argmax(dim=1)
    
    generated_sentence = ""
    for pred in prediction.numpy():
        generated_sentence += vocab_string[pred] + " "
    return generated_sentence
```
This implementation was working earlier, and I was getting a prediction of shape [11, 1293] for the question `"Was Abraham Lincoln the sixteenth President of the United States?"`. Since I didn't know what to do with this, I took the argmax of the 11, 1293 output and took string from vocab of each maximum argument. This did not work as all I got was double quotes and exclaimation marks

## Future aspirations
* To get the dataset working with latest torchtext libraries
* Get the inferencing part done right