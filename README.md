# TSAI_END2_Phase1
This repo contains the work I did during "END2.0" course for TSAI
# Assignment 8 - TorchText
> Submitted by Nikhil Shrimali

## Target
* Refer to this [Repo](https://github.com/bentrevett/pytorch-seq2seq)  

    * You are going to refactor this repo in the next 3 sessions. In the current assignment, change the 2 and 3 (optional 4, 500 additional points) such that
        * is uses none of the legacy stuff
        * It MUST use Multi30k dataset from torchtext
        * uses yield_token, and other code that we wrote
    * Once done, proceed to answer questions in the Assignment-Submission Page. 

## Submission
I have migrated the following files to the latest torchtext libraries, and same can be found in their respective jupyter notebooks
* 2 - Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.ipynb
* 3 - Neural Machine Translation by Jointly Learning to Align and Translate.ipynb
* 4 - Packed Padded Sequences, Masking, Inference and BLEU.ipynb

_Along with mentioning the usage for new libraries, I've also tried to draw differences between legacy and the newer ones_
## Dataset
We'll be using Multi30k dataset for this particular assignment. Although our scope for this assignment was limited to migrating from older libraries to newer ones, it's still good to have a look at the data we're dealing with.

The Multi30k dataset essentially contains sentences in english and german, divided into train and validation files. Each of the files has extension of the classes they represent.

Few examples from train.de file (as name suggests, train data for german `de` language)
```
1. Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.
2. Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.
3. Ein kleines Mädchen klettert in ein Spielhaus aus Holz.
4. Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster.

```

Few examples from train.en file (as name suggests, train data for english `en` language)
```
1. Two young, White males are outside near many bushes.
2. Several men in hard hats are operating a giant pulley system.
3. A little girl climbing into a wooden playhouse.
4. A man in a blue shirt is standing on a ladder cleaning a window.
```

We'll be training our network to convert german sentences to english.

### Legacy implementation: 
```
from torchtext.legacy.datasets import Multi30k
```

### That's how we do it now:
```python
from torchtext.datasets import Multi30k
```


## Tokenization and Vocabulary
### Legacy implementation : 
```python
# importing
from torchtext.legacy.data import Field, BucketIterator

# configuring spacy
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')



# tokenizing
def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

# normalizing dataset and adding sos and eos tokens
SRC = Field(tokenize=tokenize_de, 
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=True)

TRG = Field(tokenize = tokenize_en, 
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=True)

# splitting train and test data (we did this way before in our new implementation)
train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG))

# building vocab
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

```

### That's how we do it now:
```python
# importing
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# tokenizing
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en')

# yields tokens one by one so that vocab can be built from training dataset
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# defining special symbols
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# building vocab for both source language (German) and target language (English). It also normalizes the data, and define vocab for special symbols
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  # Training data Iterator 
  train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  # Create torchtext's Vocab object 
  vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln), min_freq=1, specials=special_symbols, special_first=True)
```

## Defining test and validation iterators:
### Legacy implementation : 
By using bucket iterator
```
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device)
```

### Latest implementation : 

```python
train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

test_iter = Multi30k(split='test', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
```

So easy !!

## Transformations and collation
We're just few steps away from training the dataset on defined model. We need to define something that would transform the input data into a tensor, apply augmentation if required, append `sos` and `eos` tokens at start and end of each word.

### Legacy:
Don't even ask 😕

### That's how we do it now :
```python
from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX]))
                    )

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
    vocab_transform[ln], #Numericalization
    tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch
```

Almost done now.

## Defining dataloader

### Legacy implementation :
No concept of dataloaders

### That's how we do it now :
```python
dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
```
Finally!! We're moving closer to unification of how we deal with data in vision problems and text problems 🥳🥳🥳

## Future aspirations
* Need to get in detail of each of the files and actually study what and how they're constructing their model
