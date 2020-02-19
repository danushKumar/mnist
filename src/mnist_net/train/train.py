import os
import sys
import argparse
import pickle as pkl

from mnist_net.utils import data_loader
from mnist_net.Network.Network import Network 
from mnist_net.utils.preprocess import preprocess_data

def train(layer_sizes, epoch, mini_batch_size, learning_rate,regularization, lambd):

    (x_train, y_train), (x_val, y_val) = data_loader.load_mnist_data()
    train_data, val_data = preprocess_data(x_train, y_train , x_val, y_val)
    model = Network(layer_sizes)
    model()
    model.fit(train_data, mini_batch_size=mini_batch_size, 
                          epoch=epoch, 
                          learning_rate=learning_rate,
                          validation_data=val_data,
                          regularization=regularization,
                          lambd=lambd)

    save_model(model)

def save_model(model):
    with open('model_reg.pkl', 'wb') as f:
        m = pkl.dumps(model)
        pkl.dump(m, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_sizes', nargs='+', help='number of neurons in each layer oth layer refers to input size', 
                        required=True, type=int)
    parser.add_argument('--learning_rate', help='learning rate for the optimization algorithm',
                            type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--mini_batch_size', type=int, default=10)
    parser.add_argument('--regularization', action='store_true')
    parser.add_argument('--lambd', type=float, default=0.1)
    args = parser.parse_args()
    
    train(args.layer_sizes, args.epochs, args.mini_batch_size, args.learning_rate, args.regularization, args.lambd)

if __name__ == '__main__':

    main()