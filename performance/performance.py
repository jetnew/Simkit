import numpy as np


def prob_overlap(y, y_hat, bins=10):
    """Get probabilities of each bin, ignoring zeroes."""
    p, b = np.histogram(y, bins)
    q, _ = np.histogram(y_hat, b)
    p, q = p / y.shape[0], q / y.shape[0]
    idx = np.intersect1d(np.nonzero(p), np.nonzero(q))
    return p[idx], q[idx]

def kl(p, q):
    """Compute KL divergence."""
    return np.sum(p * np.log(p / q))

def js(p, q):
    """Compute JS divergence."""
    m = 0.5 * (p + q)
    return 0.5 * (kl(p, m) + kl(q, m))