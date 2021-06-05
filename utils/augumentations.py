import random
import googletrans
from google_trans_new import google_translator as Translator

translator = Translator()
available_langs = list(googletrans.LANGUAGES.keys()) 

def back_translation(sentence):
    available_langs = list(googletrans.LANGUAGES.keys()) 
    trans_lang = random.choice(available_langs) 
    print(f"Translating to {googletrans.LANGUAGES[trans_lang]}")
    translator = Translator()
    t_text = translator.translate(sentence, lang_tgt=trans_lang) 

    en_text = translator.translate(t_text, lang_src=trans_lang, lang_tgt='en') 
    return en_text
        
def random_swap(sentence, n=5): 
    length = range(len(sentence)) 
    for _ in range(n):
        idx1, idx2 = random.sample(length, 2)
        sentence[idx1], sentence[idx2] = sentence[idx2], sentence[idx1] 
    return sentence

def random_deletion(words, p=0.5): 
    if len(words) == 1: # return if single word
        return words
    remaining = list(filter(lambda x: random.uniform(0,1) > p,words)) 
    if len(remaining) == 0: # if not left, sample a random word
        return [random.choice(words)] 
    else:
        return remaining