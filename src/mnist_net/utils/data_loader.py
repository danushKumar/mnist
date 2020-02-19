import numpy as np

from keras.datasets import mnist
from .preprocess import preprocess_data

def load_batch(batch):
    images = []
    classes = []
    
    for x, y in batch:
        images.append(x)
        class_vector = vectorize_classes(y)
        classes.append(class_vector)
    
    images = np.array(images).T
    classes = np.array(classes).T.reshape((10, -1))

    return images, classes

def load_mnist_data():
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    
    return (x_train, y_train), (x_val, y_val)

def vectorize_classes(y):
    class_vector = np.zeros((10,1))
    class_vector[y] = 1.0

    return class_vector
