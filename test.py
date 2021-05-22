import torch
import torch.nn.functional as F

def test(model, device, test_loader, sum_losses = [], mnist_losses = [], overall_loss = [], test_acc = []):
    model.eval()
    test_loss = 0
    mnist_loss = 0
    sum_loss = 0
    mnist_correct = 0
    sum_correct = 0
    with torch.no_grad():
        for (data, random_digits, target, sum_of_numbers) in test_loader:
                data, random_digits, target, sum_of_numbers = data.to(device), random_digits.to(device), target.to(device), sum_of_numbers.to(device)
                output, sum_pred = model(data, random_digits)
                mnist_loss = F.nll_loss(output, target, reduction='sum').item()
                sum_loss = F.nll_loss(sum_pred, sum_of_numbers, reduction='sum').item() # sum up batch loss
                test_loss =  mnist_loss + sum_loss
                mnist_pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                sum_pred = sum_pred.argmax(dim=1, keepdim=True)
                mnist_correct += mnist_pred.eq(target.view_as(mnist_pred)).sum().item()
                sum_correct += sum_pred.eq(sum_of_numbers.view_as(sum_pred)).sum().item()
                
    test_loss /= len(test_loader.dataset)
    mnist_loss /= len(test_loader.dataset)
    sum_loss /= len(test_loader.dataset)
    
    overall_loss.append(test_loss)
    sum_losses.append(sum_loss)
    mnist_losses.append(mnist_loss)

    print('\nTest set: Average loss: {:.4f}, Mnist Accuracy: {}/{}, Sum Accuracy: {}/{}, Total: ({:.2f}%)\n'.format(
        test_loss, mnist_correct, len(test_loader.dataset), sum_correct, len(test_loader.dataset),
        100. * ((mnist_correct + sum_correct) / (2*len(test_loader.dataset)))
        ))
    
    test_acc.append(100. * ((mnist_correct + sum_correct) / (2*len(test_loader.dataset))))