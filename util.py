import numpy as np
import scipy as sci


def normalize(vector):
    return vector/np.linalg.norm(vector)


def rollavg_convolve(a, n):
    # scipy.convolve
    assert n % 2 == 1
    return sci.convolve(a, np.ones(n, dtype='float')/n, 'same')

def change_convolve(a):
    # scipy.convolve
    return sci.convolve(a, [1,-1], 'valid')