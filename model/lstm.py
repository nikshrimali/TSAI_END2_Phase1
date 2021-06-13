import torch.nn as nn
import torch.nn.functional as F
import torch

class Lstm(nn.Module):
    
    # Define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim = 16, staggered_input = False, debug=False):
        
        super().__init__()          
        self.debug = debug
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
                if self.debug:
                    if self.staggered_input:
                        print(f"Encoder {index}")
                    else:
                        print(f"Decoder {index}")
                    print(output)
            
        dense_outputs = self.fc(torch.stack(output, dim=0) if isinstance(output, list) else output) # sending the last hidden output (which has seen everything) if encoder else sending decoder output
            
        return dense_outputs, (hidden, cell)