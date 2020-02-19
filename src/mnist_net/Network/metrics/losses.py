import numpy as np

def cross_entropy(actuals, predictions, batch_size):
    cost = -(1 / batch_size) * np.sum(np.sum(np.multiply(actuals, np.log(predictions))
                                     + np.multiply((1 - actuals), np.log(1 - predictions)), axis=0, keepdims=True))
    
    return cost