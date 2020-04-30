import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    sub1 = np.matmul(Y,np.log(P))
    diff1 = np.subtract(1,Y)
    diff2 = np.log(np.subtract(1,P))
    sub2 = np.matmul(diff1, diff2)
    result = []
    both = zip(Y,P)
    for y,p in both:
        calc = y* np.log(p) + (1 - y)* np.log(1 - p)
        #print(calc)
        result.append(calc)
    
    return -1 * np.sum(result)