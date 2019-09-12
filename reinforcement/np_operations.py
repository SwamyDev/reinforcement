import numpy as np


def reduce_mean(array):
    return np.mean(array)


def one_hot(ks, n):
    return np.array([np.eye(1, n, k=k)[0] for k in ks])


def reduce_sum(array, axis):
    return np.sum(array, axis=axis)


def log(array):
    return np.log(array).T


def softmax(roll):
    return np.exp(roll) / np.sum(np.exp(roll))
