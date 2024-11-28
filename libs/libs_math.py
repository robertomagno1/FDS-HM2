import numpy as np


def sigmoid(x):
    

    ##############################
    #We compute the sigmoid od a given input x.
    g = 1 / (1 + np.exp(-x))
    
    ##############################    

    return g

def softmax(y):

    ##############################
    #We compute softmax transformation to convert raw scores into probability distribution:

    #We calculate the exponential of input values after centering
    y_exp = np.exp(y - np.max(y, axis=1, keepdims=True))

    #Then we normalize exponential values to create a probability distribution
    softmax_scores = y_exp / np.sum(y_exp, axis=1, keepdims=True)

    ##############################
    return softmax_scores
