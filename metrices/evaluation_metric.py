from bert_score import score
from torchtext.data.metrics import bleu_score

def translate_tokens_to_sentences(output, vocab_itos, EOS_IDX, max_len=50):
    sentence = ""
    for word_pred in output.argmax(1):
        if(word_pred == EOS_IDX):
            continue
        sentence += vocab_itos[word_pred] + " "
    return sentence

def calculate_bertscore(sentence1, sentence2):
    return score([sentence1], [sentence2], lang="en", verbose=False)

def calculate_bleu_score(sentence1, sentence2):
    sentence2 = [[sentence2.split()]]
    sentence1 = [sentence1.split()]
    score = bleu_score(sentence1, sentence2)
    return score
