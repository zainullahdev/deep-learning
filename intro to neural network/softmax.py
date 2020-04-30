import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    newL = np.exp(L)
    denominator = np.sum(newL)
    return np.divide(newL,denominator)