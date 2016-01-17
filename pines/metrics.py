# coding=utf-8

import numpy as np


def list_to_discrete_rv(x):
    """
    Takes a list of positive ints and turns it into a discrete distribution
    E.g.: list_to_discrete_rv([0,0,1,1,2]) => ([0, 1, 2], [0.4, 0.4, 0.2])
    """
    unique, counts = np.unique(x, return_counts=True)
    return unique, counts / len(x)

def gini_index(x):
    if len(x) == 0:
        return 0.0
    p = list_to_discrete_rv(x)[1]
    return 1.0 - np.sum(np.power(p, 2))

def entropy(x):
    if len(x) == 0:
        return 0.0
    p = list_to_discrete_rv(x)[1]
    return -np.sum(p * np.log(p))

def mse(x):
    if len(x) == 0:
        return 0.0
    return np.std(x)