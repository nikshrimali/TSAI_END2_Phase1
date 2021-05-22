import matplotlib.pyplot as plt

def plot_loss_graphs(sum_losses, mnist_losses, overall_loss, acc, mode):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(sum_losses, label = "Sum loss")
    ax1.plot(mnist_losses, label = "Mnist digit loss")
    ax1.plot(overall_loss, label = "Overall loss")
    ax2.plot(acc, label = f"{mode} accuracy")
    ax1.set_title(f"{mode} Loss Graph")
    ax2.set_title(f"{mode} Accuracy Graph")
    ax1.legend()
    ax2.legend()
    f.savefig(f"{mode}_graphs")