import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Mnist + Sum of number options')
    parser.add_argument('--total', help='total images for inferencing', default=25)
    parser.add_argument('--epochs', help='total number of epochs for training to run', default=20)
    parser.add_argument('--lr', help='learning rate for which weights must be updated', type=float, default=0.02)
    parser.add_argument('--batch-size', help='batch size for sending data into the model', default=512)
    parser.add_argument('--save', help='save model at the end of traning', action='store_true')
    parser.add_argument('--plot-graph', help='plot training & testing loss and accuracy graphs', action='store_true')
    parser.add_argument('--summarize', help='show model summary', action='store_true')
    args = parser.parse_args()
    return args
