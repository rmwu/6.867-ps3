import numpy as np
from run_nn import onehot

def load_digit(digit):
    filename = "data/mnist_digit_{}.csv".format(digit)
    raw_data = np.loadtxt(filename)
    
    # normalize to [-1, 1] interval for better perf
    raw_data = np.apply_along_axis(lambda x : 2*x/255. - 1, 1, raw_data)
    
    return raw_data

def get_data(total, digits = None):
    """
    given a list of positive and negative numbers, returns
    a tuple (X, y) containing dictionaries of train, validate,
    and test data with an equal number of training points from 
    each digit
    
    total      total number of data points to retrieve. train,
               validate, and test data are split 2:1:1
    """
    # dictionary has keys train, validate, and test
    total_dim = 28 * 28
    
    X = {"train": np.zeros((1,total_dim)),
        "validate": np.zeros((1,total_dim)),
        "test" : np.zeros((1,total_dim))}
    
    y = {"train": np.zeros((1,10)),
        "validate": np.zeros((1,10)),
        "test" : np.zeros((1,10))}
    
    n_train = total // 2;
    n_validate = n_train + total // 4;
    
    # digits 0 to 9, inclusive, if no choices provided
    if digits is None:
        digits = range(10)
        
    for digit in digits:
        raw_data = load_digit(digit) # huge dataset
        
        # load in training data for this digit
        X["train"] = np.r_[X["train"], raw_data[:n_train]]
        y["train"] = np.r_[y["train"], np.tile(
                onehot(digit, 10), # onehot version of y
                (n_train, 1) # tile y labels vertically
            )]
        
        # load in validation data for this digit
        X["validate"] = np.r_[X["validate"], raw_data[n_train:n_validate]]
        y["validate"] = np.r_[y["validate"], np.tile(
                onehot(digit, 10), # onehot version of y
                (n_validate - n_train, 1) # tile y labels vertically
            )]
        
        # load in testing data for this digit
        X["test"] = np.r_[X["test"], raw_data[n_validate:total]]
        y["test"] = np.r_[y["test"], np.tile(
                onehot(digit, 10), # onehot version of y
                (total - n_validate, 1) # tile y labels vertically
            )]
    
    for label in ["train", "validate", "test"]:
        assert X[label].shape[0] == y[label].shape[0]
        # chop off placeholder 0s
        X[label] = X[label][1:]
        y[label] = y[label][1:]
    
    return (X, y)