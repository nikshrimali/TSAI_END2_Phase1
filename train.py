import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from test import test
from model.mnist_image_plus_random_number import Net
from torchvision import transforms
from datasets.mnist_random_digit_dataset import MnistImageWithRandomNumberDataset as MIRND

torch.manual_seed(1)
def train(epochs = 20, LR = 0.02, BATCH_SIZE = 512, device="cuda", plot = True, showSummary = True):
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        MIRND(root_directory = './data', train=True,
                    transform=[
                        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]),
        batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        MIRND(root_directory = './data', train=False, transform=[
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]),
        batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    
    model = Net().to(device)
    
    if showSummary:
        from torchsummary import summary
        summary(model, input_size=(1, 28, 28))
    
    optimizer = optim.SGD(model.parameters(), lr = LR, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    train_sum_losses = []
    train_mnist_losses = []
    train_overall_loss = []
    train_acc = []
    test_sum_losses = []
    test_mnist_losses = []
    test_overall_loss = []
    test_acc = []
    
    from tqdm import tqdm

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        pbar = tqdm(train_loader, position = 0, leave = True)
        model.train()
        mnist_correct = 0
        sum_correct = 0
        processed = 0
    
        for batch_idx, (data, random_digits, target, sum_of_numbers) in enumerate(pbar):
            # get samples
            data, random_digits, target, sum_of_numbers = data.to(device), random_digits.to(device), target.to(device), sum_of_numbers.to(device)

            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred, sum_pred = model(data, random_digits)

            # Calculate loss
            loss_digit_detection = F.nll_loss(y_pred, target)
            loss_sum_prediction = F.nll_loss(sum_pred, sum_of_numbers)
            loss = loss_digit_detection + loss_sum_prediction
            train_overall_loss.append(loss)
            train_sum_losses.append(loss_sum_prediction)
            train_mnist_losses.append(loss_digit_detection)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update pbar-tqdm
            
            mnist_pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            sum_pred = sum_pred.argmax(dim=1, keepdim=True)
            mnist_correct += mnist_pred.eq(target.view_as(mnist_pred)).sum().item()
            sum_correct += sum_pred.eq(sum_of_numbers.view_as(sum_pred)).sum().item()
            processed += len(data)
            train_acc.append(100* ((mnist_correct+sum_correct)/(2*processed)))
            pbar.set_description(desc= f'Loss={loss.item():0.2f} Batch_id={batch_idx} Mnist_Accuracy={100*mnist_correct/processed:0.2f}% Sum_Accuracy = {100*sum_correct/processed:0.2f}% Total : {train_acc[-1]:0.2f}%')
            
        scheduler.step()
        test(model, device, test_loader, test_sum_losses, test_mnist_losses, test_overall_loss, test_acc)
    
    torch.save(model.state_dict(), './mnist_and_sum_pred.pt')
        
    if plot:
        from utils.graph_plot import plot_loss_graphs
        plot_loss_graphs(train_sum_losses, train_mnist_losses, train_overall_loss, train_acc, mode="Training")
        plot_loss_graphs(test_sum_losses, test_mnist_losses, test_overall_loss, test_acc, mode="Testing")
        
        
if __name__ == '__main__':
    from utils.options import parse_args
    options = parse_args()
    print(options)
    train(epochs = int(options.epochs), LR = float(options.lr), BATCH_SIZE = int(options.batch_size), device="cuda", plot = options.plot_graph, showSummary = options.summarize)