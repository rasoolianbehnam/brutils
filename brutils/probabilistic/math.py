import numpy as np
import tensorflow as tf


def find_roots(a, b, c):
    k = b ** 2 - 4 * a * c
    if k >= 0:
        ksqr = np.sqrt(k)
    else:
        ksqr = np.sqrt(-k) * 1j

    return np.array([-b - ksqr, -b + ksqr]) / (2 * a)


def lfact(n):
    return tf.math.lgamma(n+1)


def lchoice(n, k):
    return lfact(n) - lfact(k) - lfact(n-k)