import torch
from models.transformer import Transformer
from utils.multi30k_dataset import get_train_dataloader, get_val_dataloader, get_vocab_transform
from timeit import default_timer as timer
import torch.nn as nn
import torch.optim as optim
from utils.constants import PAD_IDX, SRC_LANGUAGE, TGT_LANGUAGE

def train_epoch(model, optimizer, criterion, BATCH_SIZE):
    losses = 0
    train_dataloader = get_train_dataloader(BATCH_SIZE)
    
    for src, tgt in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt)
        
        output = output[1:].view(-1, output.shape[-1])
        tgt = tgt[1:].reshape(-1)
        loss = criterion(output, tgt)
        loss.backward()
        clip = 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)

def evaluate(model, BATCH_SIZE):
    model.eval()
    losses = 0
    val_dataloader = get_val_dataloader(BATCH_SIZE)
    
    for src, tgt in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        
        output = model(src, tgt)
        output = output[1:].view(-1, output.shape[-1])
        tgt = tgt[1:].reshape(-1)
        loss = criterion(output, tgt)
        losses += loss.item()

    return losses / len(val_dataloader)

if __name__ == "__main__":
    
    NUM_EPOCHS = 1
    BATCH_SIZE = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_transform = get_vocab_transform()
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TRG_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    model = Transformer(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, PAD_IDX, PAD_IDX, device=device).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(model, optimizer, criterion, BATCH_SIZE)
        end_time = timer()
        val_loss = evaluate(model, BATCH_SIZE)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))