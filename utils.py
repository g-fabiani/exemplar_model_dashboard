import numpy as np


def activation(x, point,  alpha):
    dist = x-point
    return np.exp(-dist**2 / (2 * alpha**2))


def discriminability(ratio, delta):
    return ratio / (ratio + delta)


def typicality(avg_activation, tau):
    return 1 - (1 / 2 ** (avg_activation/tau))
