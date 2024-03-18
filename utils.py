import numpy as np


def activation(x, point,  alpha):
    dist = x-point
    return np.exp(-dist**2 / (2 * alpha**2))


def discriminability(ratio, delta):
    return ratio / (ratio + delta)


def typicality(avg_activation, tau):
    return 1 - (1 / 2 ** (avg_activation/tau))


def set_delta(freq, lambda_param=.25, phi=.5):
    M = 12
    delta = lambda_param - (((2 * (freq - 1) / (M - 1)) - 1) * phi)
    return min(max(delta, 0), 1)


map_delta_freq = {freq: set_delta(freq)
                  for freq in range(1, 13)}
