# -*- coding: utf-8 -*-

from utils import get_device, seed_all
from models import get_latest_checkpoint, GetEncodings, BARTTrain

seed_all()
device = get_device()
# Commented out IPython magic to ensure Python compatibility.
# BERT imports

# % matplotlib inline

import faiss




# specify GPU device

df_context = pd.read_csv('/content/drive/MyDrive/END2_CAPSTONE/context_groups.csv') 
df_merged = pd.read_csv('/content/drive/MyDrive/END2_CAPSTONE/merged_data.csv')
model_op = torch.load('/content/drive/MyDrive/END2_CAPSTONE/np_encoded_context.pt')




# Inference
MODEL_STORE = '/content/drive/MyDrive/END2_CAPSTONE'

bart_model = BARTTrain().to(device)



bart_model = get_latest_checkpoint('checkpoint', bart_model, MODEL_STORE)

def inference(question, bart_tokenizer, bart_model):

    # Get Pretrained BERT encodings

    ge = GetEncodings(type='questions')
    encoded_question = ge.encode(question, max_length=30)

    # Find top matching documents
    ss = SearchSimilar(iterator = df_context['context'].values.tolist(), filename='index.bin', embeddings=model_op, shape=768, device=device)
    similar_contexts = ss.get_n_similar_vectors(encoded_question, 3)
    similar_contexts.insert(0, question)

    combined_tokens = '</s></s>'.join(similar_contexts)

    print(f'Top similar document outputs is {combined_tokens}')

    # Prepare data for BART Inferencing

    source_encoding = tokenizer(
            combined_tokens,
            max_length=1024,
            padding='max_length',
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt")
   

    # Inference BART Model
    output = bart_model(source_encoding['input_ids'].to(device), mode = 'eval')
    output = tokenizer.decode(output[0])
    print(output)
    return output

tokens = inference('What does torch.cosine loss do in pytorch?', tokenizer, bart_model)

# Loss plots


loss_qna = torch.load('/content/drive/MyDrive/END2_CAPSTONE/qna_checkpoint-4000/training_loss.pt')
