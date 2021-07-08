import numpy as np
from scipy.stats import beta


def weighted_quantile(x, w, p):
    n = np.sum(w) / np.max(w)
    a, b = p * (n + 1.), (1. - p) * (n + 1.)

    sort_indexes = np.argsort(x)
    x_sorted, w_sorted = x[sort_indexes], w[sort_indexes]
    w_sorted /= np.sum(w_sorted)

    probs = np.cumsum(np.concatenate(([0.], w_sorted)))

    W = beta.cdf(probs, a, b)

    return (W[1:] - W[:-1]) @ x_sorted
