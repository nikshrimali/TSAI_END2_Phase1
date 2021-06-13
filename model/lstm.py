import torch.nn as nn
import torch.nn.functional as F
import torch

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
        
    def forward(self, source_text, num_steps, hidden, cell):
        # send embeddings inside input for encoder, and encoder output for decoder. Also, try to send packed embedded inside input too
        
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
            
        dense_outputs = self.fc(torch.stack(output, dim=0) if isinstance(output, list) else output) # sending the last hidden output (which has seen everything) if encoder else sending decoder output
            
        return dense_outputs, (hidden, cell)