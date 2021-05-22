import torch
from model.mnist_image_plus_random_number import Net
from torchvision import transforms
from datasets.mnist_random_digit_dataset import MnistImageWithRandomNumberDataset as MIRND
import matplotlib.pyplot as plt
import numpy as np

def inference(model= None, total=25):
    data_loader = torch.utils.data.DataLoader(
        MIRND(root_directory = './data', train=False, transform=[
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]),
        batch_size=total, shuffle=True)
    plt.figure(figsize=(15, 20))
    
    if model is None:
        model = Net()
        model.load_state_dict(torch.load("./mnist_and_sum_pred.pt"))
    with torch.no_grad():
        data, random_digits, target, sum_of_numbers = next(iter(data_loader))
        mnist_output, sum_pred = model(data, random_digits)
        sum_pred = sum_pred.argmax(dim=1, keepdim=True).numpy()
        random_digits = random_digits.argmax(dim=1, keepdim=True).numpy()
        mnist_output = mnist_output.argmax(dim=1, keepdim=True).numpy()
        
        for image_num in range(mnist_output.shape[0]):
            valid = False
            if sum_of_numbers[image_num] == sum_pred[image_num][0]:
                valid = True
            plt.subplot(5, 5, image_num+1)
            plt.axis('off')
            plt.imshow(np.transpose(data[image_num], (1, 2, 0)))
            plt.title(f'Target: {target[image_num]} + {sum_of_numbers[image_num] - target[image_num]} = {sum_of_numbers[image_num]}. \n Prediction: {mnist_output[image_num][0]} + {sum_of_numbers[image_num] - target[image_num]}  = {sum_pred[image_num][0]} \n Valid = {valid}')
            
    plt.savefig("Inferencing_Mnist_Sum_Of_Numbers")
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Mnist + Sum of number inferencing options')
    parser.add_argument('--total', help='total images for inferencing', default=25)
    args = parser.parse_args()

    inference(total=int(args.total))