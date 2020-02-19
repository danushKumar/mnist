import numpy as np

from mnist_net.utils.activations import sigmoid
from mnist_net.utils.data_loader import load_batch
from .metrics.losses import cross_entropy

class Network(object):

    def __init__(self, layer_sizes):

        self.input_size = layer_sizes[0]
        self.layer_sizes = layer_sizes
        self.weights = {}
        self.bias = {}

    def __call__(self):
        
        self.initialize_weights()


    def initialize_weights(self):
        
        self.weights = {i + 1: np.random.randn(shape[0], shape[1]) 
                            for i, shape in enumerate(zip(self.layer_sizes[1 : ], self.layer_sizes[ : -1]))}
        self.bias = {i + 1: np.random.randn(size, 1) 
                        for i, size in enumerate(self.layer_sizes[1 : ])}
    

    def forward(self, x, optimizer=True):

        assert(x.shape[0] == self.input_size)
        
        A = {}
        Z = {}
        A[0] = x.copy()

        for i in range(len(self.weights)):
            Z[i + 1] = np.dot(self.weights[i + 1], A[i]) + self.bias[i + 1]
            A[i + 1] = sigmoid(Z[i + 1])
        if optimizer:
            return A 
        
        return A[len(self.weights)]
    
    def fit(self, train_data, mini_batch_size=None, epoch=100, 
                learning_rate=0.001, validation_data=None, *, regularization=False, lambd=0.1):

        self.sgd(train_data, mini_batch_size, epoch, learning_rate, validation_data, regularization, lambd)

    def sgd(self, train_data, mini_batch_size, epoch, learning_rate, validation_data, regularization, lambd):

        batches = None
        m = mini_batch_size if mini_batch_size else len(train_data)

        for i in range(epoch):
            np.random.shuffle(train_data)
            batches = [train_data[i: i + m] for i in range(0, len(train_data), m)]

            for batch in batches:
                (x, y) = load_batch(batch)
                activation_cache = self.forward(x)
                self.update_parameters(activation_cache, m, y, learning_rate, regularization, lambd)

            
            self.training_cost, self.training_accuracy = self.cost_and_accuracy_calculator(batches, 
                                                    len(train_data), 
                                                    lambd=lambd if regularization else 0) 

            print(f'epoch {i}  training_acc: {self.training_accuracy:16} training_loss: {self.training_cost}', end=' ')

            if validation_data:
                validation_batches = [validation_data[i: i + m] for i in range(0, len(validation_data), m)]
                self.validation_cost, self.validation_accuracy = self.cost_and_accuracy_calculator(validation_batches, 
                                                                        len(validation_data), 
                                                                        lambd=lambd if regularization else 0)
                                                                        
                print(f'validation_acc:{self.validation_accuracy} validation_loss: {self.validation_cost}')
    
    def update_parameters(self, activations, mini_batch_size, y, learning_rate, regularization, lambd):

        dw, db = self.backpropagate(activations, y, mini_batch_size, regularization, lambd)
        L = len(self.weights)

        assert(L == len(dw))

        for l in range(L):
            self.weights[l + 1] = self.weights[l + 1] - (learning_rate * dw[l + 1])
            self.bias[l + 1] = self.bias[l + 1] - (learning_rate * db[l + 1]) 
 
    def backpropagate(self, activations, y, mini_batch_size, regularization, lambd):
        
        L = len(self.weights)
        
        regularization_parameter = (lambd / mini_batch_size) if regularization else 0
        regularization_effect =  regularization_parameter * self.weights[L] 
        
        dz = {}
        dw = {}
        db = {}
        da = -(np.divide(y, activations[L]) - np.divide((1 - y), (1- activations[L])))

        dz[L] = da * self.sigmoid_derivative(activations[L])
        dw[L], db[L] = self.compute_derivatives(dz[L], activations[L - 1], mini_batch_size, regularization_effect)
        da = np.dot(self.weights[L].T, dz[L] )

        for l in reversed(range(1, L)):
            dz[l] = da * self.sigmoid_derivative(activations[l])

            regularization_effect = regularization_parameter * self.weights[l]
            dw[l], db[l] = self.compute_derivatives(dz[l], activations[l - 1], 
                                mini_batch_size, 
                                regularization_effect)
            da = np.dot(self.weights[l].T, dz[l])
        
        return dw, db 
    
    def compute_derivatives(self, dz, a_prev, mini_batch_size, regularization_effect):

        dw = (1 / mini_batch_size) * np.dot(dz, a_prev.T) + regularization_effect 
        db = (1 / mini_batch_size) * np.sum(dz, axis=1, keepdims=True)

        return dw, db
    
    def cost_and_accuracy_calculator(self, batches, n, *, lambd=0):
        cost = 0
        correct_preds = 0

        for batch in batches:
            (x, y) = load_batch(batch)
            preds = self.forward(x, False)
            cost += cross_entropy(y, preds, n)
            correct_preds += self.evaluate(preds, y)
        
        accuracy = (correct_preds / n) * 100
        
        if not lambd:
            return cost, accuracy 
            
        weights_norm = sum(np.linalg.norm(self.weights[l + 1])**2 for l in range(len(self.weights)))
        regularization_cost = (lambd / (2 * n)) * weights_norm
        cost += regularization_cost

        return cost, accuracy
    
    def evaluate(self, preds, y):
        
        assert(preds.shape == y.shape)

        preds = np.argmax(preds, axis=0)
        y = np.argmax(y, axis=0)
        preds = preds.reshape(1, -1)
        y = y.reshape(1, -1)

        return np.sum(preds == y)

    def sigmoid_derivative(self, a):

        return a * (1-a)
        