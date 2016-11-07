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
    weights = train(X, y, network_size, iterations, relu, relu_grad, softmax_grad)
    
    print("Converged to weight vector {} after {} iterations".format(weights, iterations))
    return weights

def train(X, y, network_size, max_iterations,
          activation_func, activation_grad, output_func):
    """
    X       n by d input data
    y       n by 1 data labels

    network_size    list containing number of nodes per layer, including
                    the input layer's size, excluding output layer
    max_iterations  maximum number of iterations

    activation_func activation function for hidden layers
    activation_grad activation gradient for hidden layers
    output_grad     activation gradient for output layer
    """
    weights, bias = initialize(network_size) # jagged list

    n, d = X.shape
    assert len(y.shape) == 1 # want 1-d vector ONLY

    time = 0
    while time < max_iterations:
        i = np.random.randint(0, n) # choose random sample
        xi, yi = X[i], y[i]

        learning_rate = 1 / time**2 # decaying learning rate
        # jagged list, same dimensions as weights
        del_weights, del_bias = learn(xi, yi, weights, bias
                    activation_func, acivation_grad, output_grad)

        # check sizes match, and update
        assert len(weights) == len(del_weights) == len(del_bias)
        for k in range(len(weights)):
            weights[k] = weights[k] - learning_rate * del_weights[k]
            bias[k] = bias[k] - learning_rate * del_bias[k]

    return (weights, bias)

##################################
# Mathematical implementations
##################################

def relu(x):
    """
    returns the ReLU of vector x, which is d by 1
    """
    return np.maximum(x, 0, x)

def relu_grad():
    """
    returns a FUNCTION that finds the gradient of ReLU
    at vector x, where we set the subgradient at 0 to 0
    """
    def gradient(x):
        grad = []
        for xi in x:
            g = 1 if x > 0 else 0
            grad.append(g)
        return grad
    return np.vectorize(gradient)

def softmax(x):
    """
    returns the softmax of vector x
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis = 0)

def softmax_grad(x):
    """
    returns the softmax of vector x
    """
    # TODO
    return x

def cross_entropy(fz):
    """
    returns a FUNCTION for the cross entropy for point (x, y),
    which is defined as

        \sum_i {y_i \log{f(z)_i}}

    where f(z) is the output layer output 
    """
    def loss(x, fz):
        return sum([x_i + math.log(fz_i)
                for x_i, fz_i in list(zip(x, fz))])

    return lambda x : np.vectorize(loss)(x, fz)

def cross_entropy_grad(fz):
    """
    returns a FUNCTION for the cross entropy gradient for point (x, y),
    which would be
        [ - y_i / fz_i ] for y_i \in y
    """
    return lambda x : - x / fz

##################################
# Command line
##################################

if __name__ == "__main__":
    main()