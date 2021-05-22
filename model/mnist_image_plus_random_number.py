import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convLayer1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),            
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, 1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )
        self.convLayer2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, bias=False),            
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, 1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)

        )
        self.convLayer3 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, bias=False),            
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, 1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )
        self.convLayer4 = nn.Sequential(
            nn.Conv2d(8, 16, 3, bias=False),
            nn.Conv2d(16, 10, 1, bias=False),
        )
        self.fcSumLayer1 = nn.Sequential(
            nn.Linear(in_features = 20, out_features=20, bias = False),
            nn.BatchNorm1d(20),
            nn.ReLU()
        )
        
        self.fcSumLayer2 = nn.Linear(in_features = 20, out_features=19, bias = False)


    def forward(
        self, x, 
        random_digit = torch.tensor(np.zeros(20), device="cuda").view(-1, 10).float() # default value only for torchsummary, not to be passed into actual model 
        ):
        x = self.convLayer1(x)
        x = self.convLayer2(x)
        x = self.convLayer3(x)
        x = self.convLayer4(x)
        x = x.view(-1, 10)
        y = self.fcSumLayer1(
            torch.cat(
                (
                    x, 
                    random_digit
                )
                , dim=1)
            )
        y = self.fcSumLayer2(y)
        y = y.view(-1, 19)
        return F.log_softmax(x, dim=-1), F.log_softmax(y, dim=-1)
    
    def _make_one_hot_encoding_tensor(number):
        array = np.zeros(10)
        array[number] = 1
        return torch.tensor([array], device="cuda")