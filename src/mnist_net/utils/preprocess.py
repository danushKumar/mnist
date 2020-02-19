import numpy as np

def preprocess_data(x_train, y_train, x_test, y_test):
    x_train = rescale(x_train)
    x_test = rescale(x_test)

    x_train = x_train.reshape((-1, 28*28))
    x_test =  x_test.reshape((-1, 28*28))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    train_data = list(zip(x_train, y_train))
    test_data = list(zip(x_test, y_test))

    return train_data, test_data

def rescale(data):
    rescaled_image = []

    for x in data:
        rescaled = x / 255.0
        rescaled_image.append(rescaled)
    
    rescaled_image = np.array(rescaled_image) 
    print(rescaled_image.shape)
    
    return rescaled_image
