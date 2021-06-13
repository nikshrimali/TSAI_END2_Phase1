import torch 
import torch.nn as nn

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
        
        return self.linear_layer(decoder_output)