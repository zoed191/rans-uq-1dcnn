import numpy as np


def moving_average_1d(a, window_size):
    assert window_size > 0, 'window_size needs to be positive'
    return np.convolve(np.pad(a, window_size-1, mode='edge'), np.ones(window_size), 'valid')[window_size-1:window_size-1+len(a)] / window_size


