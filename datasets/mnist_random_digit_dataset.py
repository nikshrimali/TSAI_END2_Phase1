import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import random
import numpy as np

class MnistImageWithRandomNumberDataset(Dataset):
    def __init__(self, root_directory, train=False, transform=None):
        self.mnist_dataset = datasets.MNIST(root = root_directory, train=train, download=True, transform=transforms.Compose(transform))
        
    def __len__(self):
        return self.mnist_dataset.__len__()
        
    def __getitem__(self, index: int):
        image, label = self.mnist_dataset.__getitem__(index)
        random_number = random.randint(0, 9)
        one_hot = np.zeros(10)
        one_hot[random_number] = 1
        random_digit_one_hot_encoding = torch.tensor(one_hot).float()
        random_digit_one_hot_encoding[random_number] = 1.
        return image, random_digit_one_hot_encoding, label, random_number + label
