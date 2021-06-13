import random
import googletrans
from google_trans_new import google_translator as Translator
import pandas as pd
import random
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords
import string
stopwords = set(stopwords.words('english') + ['AT_USER','URL'])
translator = Translator()
available_langs = list(googletrans.LANGUAGES.keys()) 
transformation_list=['all', 'back_trans', 'random_swap', 'random_deletion']
    
def processTweet(tweet):
    # tweet is the text we will pass for preprocessing
    # convert passed tweet to lower case 
    tweet = str(tweet).lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
    tweet = re.sub(r'#([\s]+)', r'\1', tweet) # remove the # space
    tweet = tweet.replace("'", "")
    tweet = re.sub('[^A-Za-z0-9]+', ' ', tweet) # remove # and numbers
    # use work_tokenize imported above to tokenize the tweet
    tweet =  word_tokenize(tweet)
    return ' '.join([word for word in tweet if word not in stopwords or word not in list(punctuation)])

def back_translation(sentence):
    available_langs = list(googletrans.LANGUAGES.keys()) 
    trans_lang = random.choice(available_langs) 
    print(f"Translating to {googletrans.LANGUAGES[trans_lang]}")
    translator = Translator()
    t_text = translator.translate(sentence, lang_tgt=trans_lang) 

    en_text = translator.translate(t_text, lang_src=trans_lang, lang_tgt='en') 
    return en_text
        
def random_swap(sentence, n=2):
    sentence = list(sentence) 
    length = range(len(sentence))
    for _ in range(n):
        idx1, idx2 = random.sample(length, 2)
        sentence[idx1], sentence[idx2] = sentence[idx2], sentence[idx1]
    return ''.join(sentence)

def random_deletion(words, p=0.1): 
    if len(words) == 1: # return if single word
        return words
    remaining = list(filter(lambda x: random.uniform(0,1) > p,words)) 
    if len(remaining) == 0: # if not left, sample a random word
        converted_list =  [random.choice(words)]
        return "".join(map(str, converted_list))
    else:
        return "".join(map(str, remaining))

def apply_augmentations(tweet, label, p = 0.2, transforms = 'all', colummn_names = ["tweets", "labels"]):
    if transforms not in transformation_list:
        raise RuntimeError(f"The specified transforms ({transforms}) could not be found. Current available transforms are : 1. all - applies everything, 2. back_trans, 3. random_swap, 4. random_deletion")
    
    series = pd.Series([])
    # if transforms == 'all' or transforms == 'back_trans': 
    #     if random.uniform(0, 1) <= p:
    #         series = series.append(
    #             pd.Series(
    #                 [
    #                     back_translation(tweet),
    #                     label
    #                 ],
    #                 index = colummn_names
    #             )
    #         )
    
    if transforms == 'all' or transforms == 'random_swap': 
        if random.uniform(0, 1) <= p:
            series = series.append(
                pd.Series(
                    [
                        random_swap(tweet),
                        label
                    ],
                    index = colummn_names
                )
            )
    if transforms == 'all' or transforms == 'random_deletion': 
        if random.uniform(0, 1) <= p:
            series = series.append(
                pd.Series(
                    [
                        random_deletion(tweet),
                        label
                    ],
                    index = colummn_names
                )
            )
    return series