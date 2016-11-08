import numpy as np
import math
from collections import deque

def test_nn(X, y, weights, bias, activation_func, output_func, loss_func):
    """
    X       n by d input data
    y       n by 1 data labels
    
    weights         jagged weight array for all layers from 2 to L
    bias            bias vector for all layers from 2 to L, length L-1

    activation_func activation function for hidden layers
    output_func     activation function for output layer
    loss_func       loss function after output
    """
    guesses = []
    losses = []
    
    for i in range(len(X)):
        z, activations = forward_prop(X[i], weights, bias, activation_func, output_func)
        softmax = activations[-1]
        
        error_func = loss_func(activations[-1])
        loss = error_func(y[i])
        losses.append(loss)
        
        guess = np.rint(softmax) # adjust to one-hot
        guesses.append(guess)
        # print((softmax, guess))
                
    correct = 0
    for i in range(len(guesses)):
        if (guesses[i] == y[i]).all():
            correct += 1
            
    ratio = correct / len(X)
    loss = np.mean(losses)
    
    print("Guessed {} correct out of {} ({})\nAverage loss {}\n".format(correct, len(X), ratio, loss))
    return (loss, ratio)

def train_nn(X, y, network_size, max_iterations,
          activation_func, activation_grad, 
          output_grad, output_func, loss_grad, loss_func):
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
    assert y.shape[0] == X.shape[0] # must line up

    time = 1
    last_loss = 0
    
    while time <= max_iterations:
        i = np.random.randint(0, n) # choose random sample
        xi, yi = X[i], y[i]
        
        # print("Training on x = {} and y = {}\n".format(xi, yi))        
        learning_rate = 0.01
        
        # jagged list, same dimensions as weights
        del_weights, del_bias = learn(xi, yi, weights, bias,
                activation_func, activation_grad, 
                output_grad, output_func, loss_grad)
        # print("del weights {}\n del bias {}\n".format(del_weights, del_bias))

        # check sizes match, and update
        assert len(weights) == len(del_weights) == len(del_bias)
        for k in range(len(weights)):
            weights[k] = weights[k] - learning_rate * del_weights[k].T
            bias[k] = bias[k] - learning_rate * del_bias[k]
            
        # weights, _ = initialize(network_size) # try just making it random again
        
        # evaluate loss every 500 turns
        if time % 500 == 0:
            # print((np.linalg.norm(del_weights[0]), np.linalg.norm(del_bias[0])))
            last_loss, end = terminate(X, y, weights, bias, last_loss,
                         activation_func, output_func, loss_func)
            if end:
                break
        
        # gg don't forget to increment this...
        time += 1

    print("Ran for {} iterations.".format(time))
    return (weights, bias)

def terminate(X, y, weights, bias, last_loss,
              activation_func, output_func, loss_func):
    loss, ratio = test_nn(X, y, weights, bias, activation_func, output_func, loss_func)
    if ratio > 0.99:
        return (loss, True)
    if abs(last_loss - loss) < 0.0001:
        return (loss, True)
    return (loss, False)

def initialize(network_size):
    """
    returns an initial list of weight arrays, initialize to normal distributions
    around 0 with stdev dependent on size of layer

    network_size    list containing number of nodes per layer, including
                    the input layer's size, excluding output layer
    """
    weights = []
    bias = []

    # we "look behind" to get proper sizes
    prev_n = network_size[0]
    for i in range(1, len(network_size)):
        n = network_size[i]
        
        stdev = 1 / n # normal stdev decreases as number of nodes in layer

        # initialize weights around 0 with gaussian dist
        weights_init = np.random.normal(loc = 0, scale = stdev, 
                            size = n * prev_n).reshape(n, prev_n)

        # add the newly generated weights
        weights.append(weights_init)
        
        # add the newly generated bias
        bias_init = np.random.normal(loc = 0, scale = stdev, size = n)
        bias.append(bias_init)

        prev_n = n # update previous
                
    return (weights, bias)

def learn(xi, yi, weights, bias, activation_func, activation_grad, 
          output_grad, output_func, loss_grad):
    """
    learn implements the backpropogation algorithm for finding gradients with
    respect to parameters

    X       n by d input data
    y       n by 1 data labels

    weights         jagged weight array for all layers from 2 to L
    bias            bias vector for all layers from 2 to L, length L-1

    activation_func activation function for hidden layers
    activation_grad activation gradient for hidden layers
    output_grad     activation gradient for output layer
    loss_grad       gradient of loss function
    """
    z, activations = forward_prop(xi, weights, bias, activation_func, output_func)
    # print("z {}\n activations {}\n".format(z, activations))
    
    grad_weights, grad_bias = back_prop(z, yi, activations, weights,
                            activation_grad, output_grad, output_func, loss_grad)
    
    return (grad_weights, grad_bias)

def forward_prop(xi, weights, bias, activation_func, output_func):
    """
    forward_prop returns the activations from all layers

    xi              single input vector, d by 1

    weights         jagged weight array for all layers from 2 to L
    bias            bias vector for all layers from 2 to L, length L-1
    activation_func activation function for hidden layers
    """
    # print("W length {} and bias length {}".format(len(weights), len(bias)))
    assert len(weights) == len(bias) # check that sizes align

    activations = [xi] # input layer activations
    weighted_inputs = []
        
    # start at 1, since we stick in the activations for input layer
    for i in range(0, len(weights)):
        # print("bias_{} = {}".format(i, bias[i]))
        w, b = weights[i], bias[i]
        prev_activation = activations[i]

        # now calculate weighted input and output function
        z = w.dot(prev_activation) + b
        
        # print("activation {} \n weight {} \n bias {}\n z {} \n"\
        #      .format(prev_activation, weight, bias, z))

        weighted_inputs.append(z)        
        activations.append(activation_func(z))

    # adjust for softmax
    # print(weighted_inputs[-1])
    activations[-1] = output_func(weighted_inputs[-1])
    return (weighted_inputs, activations)

def back_prop(z, yi, activations, weights, 
              activation_grad, output_grad, output_func, loss_grad):
    """
    z           weighted input vectors for each layer
    activations activations by the activation function
    weights     jagged weight array for all layers from 2 to L

    activation_grad gradient function of activation
    output_grad     gradient function of output layer
    loss_grad       gradient of loss function (used at the end)
    """
    L = len(activations) - 1

    # initialize error
    deltas = deque()
    # set output layer error
    last_delta = activations[-1] - yi
    deltas.append(last_delta)

    # L-1 down to 2, shifted by -2
    for i in range(L-2, -1, -1):
        # most recently added should be leftmost
        
        # print("weights {}\n zi {}\n delta {}\n".format(z[i], weights[i+1], deltas[0]))
        delta = (activation_grad(z[i]) * weights[i + 1]).T.dot(deltas[0])
        deltas.appendleft(delta)

    # print("deltas {}\n".format(deltas))
    # calculate gradients
    assert len(activations)-1 == len(deltas)
    
    grad_weights = []
    for i in range(L):
        # print("activation {}\n delta {}\n".format(activations[i], deltas[i]))
        delta = deltas[i].reshape(len(deltas[i]), 1)
        activation = activations[i].reshape(len(activations[i]), 1)
        
        grad_weights.append(activation.dot(delta.T))
    grad_bias = deltas

    return (grad_weights, grad_bias)
