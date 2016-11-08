import numpy as np
import math
from nn import *

"""
run_nn provides concrete mathematical implementations
for our neural network
"""

##################################
# Running the neural network
##################################

def main():
    if len(sys.argv) != 2:
        print("Usage: %s data_file" % sys.argv[0])
        sys.exit(1)

    # parameters
    filename = sys.argv[1]
    iterations = 1000

    train = np.loadtxt(filename)
    X = train[:,0:2]
    y = train[:,2:3]

    assert X.shape[1] == y.shape[0] # check dimensions

    n, _ = y.shape # make into 1-d vector
    y = y.reshape(n,)

    # neural network construction
    network_size = [2, 3] # list containing nodes per hidden layer
    weights = train_nn(X, y, network_size, iterations, relu, relu_grad, softmax_grad, cross_entropy_grad)
    
    print("Converged to weight vector {} after {} iterations".format(weights, iterations))
    return weights

##################################
# Mathematical implementations
##################################

def relu(x):
    """
    returns the ReLU of vector x, which is d by 1
    """
    return np.maximum(x, 0)

def relu_grad(x):
    """
    returns the gradient of ReLU at vector x, where we set 
    the subgradient at 0 to 0
    """
    grad = []
    for xi in x:
        g = 1 if xi > 0 else 0
        grad.append(g)
    return np.array(grad)

def softmax(x):
    """
    returns the softmax of vector x
    """
    e_x = np.exp(x - np.max(x)) # normalization
    return e_x / e_x.sum(axis = 0) # divide by magnitude

def softmax_grad(y, i):
    """
    returns the gradient of softmax of y w.r.t. z?
    """
    n = len(y)
    y = y.reshape(n, 1)
    yt = y.reshape(1, n)
    
    grad = y.dot(yt)
    
    for i in range(n):
        grad[i][i] = y[i] * (1 - y[i])
    
    return grad

def cross_entropy(fz):
    """
    returns a FUNCTION for the cross entropy for point (x, y),
    which is defined as

        \sum_i {y_i \log{f(z)_i}}

    where x is a MATRIX
    and f(z) is the output layer output 
    """
    def loss(x, fz):
        assert len(x) == len(fz)
        
        tot_loss = 0
        for i in range(len(x)):
            tot_loss += x[i] * math.log(fz[i])
        return -tot_loss
    return lambda x : loss(x, fz)

def cross_entropy_grad(fz):
    """
    returns a FUNCTION for the cross entropy gradient for point (x, y),
    which would be
        [ - y_i / fz_i ] for y_i \in y
    where y_i is a VECTOR
    """
    return lambda y : fz - y # or the other way around

def onehot(y, n):
    """
    y      class label, indexed from 0
    n      number of dimensions
    """
    vector = np.zeros(n)
    vector[y] = 1
    return vector

##################################
# Command line
##################################

if __name__ == "__main__":
    main()