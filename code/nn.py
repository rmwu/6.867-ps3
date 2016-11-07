import numpy as np
import math

def initialize(network_size):
    """
    returns an initial list of weight arrays, initialize to normal distributions
    around 0 with stdev dependent on size of layer

    network_size    list containing number of nodes per layer, including
                    the input layer's size, excluding output layer
    """
    weights = []
    bias = np.zeros(len(network_size))

    # we "look behind" to get proper sizes
    prev_n = network_size[0]
    for n in range(1, len(network_size), 1):
        stdev = 1 / n # normal stdev decreases as number of nodes in layer

        # initialize weights around 0 with gaussian dist
        weights_init = np.random.normal(loc = 0, scale = stdev, 
                                size = n * prev_n).reshape(n, prev_n)

        # add the newly generated weights
        weights.append(weights_init)
        # update the bias term with this stdev
        bias[n] = np.asscalar(np.random.normal(loc = 0, scale = stdev, size = 1))

        prev_n = n # update previous

    return (weights, bias)

def learn(xi, yi, weights, bias, activation_func, activation_grad, output_grad):
    """
    X       n by d input data
    y       n by 1 data labels

    weights         jagged weight array for all layers from 2 to L
    bias            bias vector for all layers from 2 to L, length L-1

    activation_func activation function for hidden layers
    activation_grad activation gradient for hidden layers
    output_grad     activation gradient for output layer
    """
    z, activations = forward_prop(xi, weights, bias, activation_func)
    grad_weights, grad_bias = back_prop(z, activations, weights, activation_grad, output_grad)

def forward_prop(xi, weights, bias, activation_func):
    """
    forward_prop returns the activations from all layers

    xi              single input vector, d by 1

    weights         jagged weight array for all layers from 2 to L
    bias            bias vector for all layers from 2 to L, length L-1
    activation_func activation function for hidden layers
    """
    assert weights.shape[0] == bias.shape[0] # check that sizes align

    activations = [xi]
    weighted_inputs = [] # TODO figure out dimensions

    # start at 1, since we stick in the activations for input layer
    for i in range(1, weights.shape[0], 1)
        weight, bias = weights[i], bias[i]
        prev_activation = activations[i - 1]

        # now calculate weighted input and output function
        z = weight.T.dot(prev_activation) + bias

        weighted_inputs.append(z)
        activations.append(activation_func(z))

    return (weighted_inputs, activations)

def back_prop(z, activations, weights, activation_grad, output_grad):
    """
    z           weighted input vectors for each layer
    activations activations by the activation function
    weights     jagged weight array for all layers from 2 to L

    activation_grad gradient function of activation
    output_grad     gradient function of output layer
    """
    L = len(activations) - 1

    # initialize error for output layer
    deltas = np.zeros(L - 1)
    deltas[L - 2] = diag([output_grad(z[L])]).dot(delsomething())

    # L-1 down to 2, shifted by -2 lol
    for i in range(L - 3, -1, 1):
        delta = diag(activation_grad(z[i])).dot(weights[i+1]).dot(delta(i+1))
        deltas[i] = delta

    # calculate gradients
    assert len(activations) == len(deltas)
    grad_weights = activations.dot(deltas.T)
    grad_bias = delta

    return (grad_weights, grad_bias)

def diag(f):
    """
    TODO figure out what diag does
    """
    return f

def delsomething():
    pass
