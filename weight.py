import numpy as np


def weight(x: object, xi: object, t: object) -> object:
    """returns the weight matrix based from x with parameter and t"""
    x_av = np.average(x)
    w = np.exp(-(x - xi)*(x - xi)/(2*t*t))
    wmatrix = np.diag((w.reshape(450)))
    return wmatrix

