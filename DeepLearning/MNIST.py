import numpy as np
import gzip
import pickle
import random

def get_dataset(path_zip):
    tr_d, va_d, te_d = load_data(path_zip) # initialization of datasets in MNIST format
    train_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    train_results = [vectorized_result(y) for y in tr_d[1]]
    train_data = zip(train_inputs, train_results)
    valid_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    valid_data = zip(valid_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    return (train_data, valid_data, test_data)

def get_test_image(path_zip):
    _, _, te_d = load_data(path_zip)  # initialization of datasets in MNIST format
    images=te_d[0]
    random_index=random.randint(0,len(images))
    random_image=images[random_index]
    return np.reshape(random_image, (28, 28))


def load_data(path_zip):
    f = gzip.open(path_zip, 'rb')
    train_data, valid_data, test_data = pickle.load(f,encoding='latin1')
    f.close()
    return (train_data, valid_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e