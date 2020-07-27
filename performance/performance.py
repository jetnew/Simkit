import numpy as np


def prob_overlap(y, y_hat, bins=5):
    """Get probabilities of each bin, ignoring zeroes.
    
    Computes the probabilities of multivariate histogram bins,
    for the computation of empirical KL and JS divergences.
    
    Note: Ignores bins for both p and q if either p or q equals 0,
    resulting in sometimes invalid computation of KL and JS divergences.
    Either model smaller dimensions or improve the fit of the model.
    """
    p, b = np.histogramdd(y, bins)
    q, _ = np.histogramdd(y_hat, b)
    p, q = p / y.shape[0], q / y.shape[0]
    idx = np.array(np.where(np.logical_and(p != 0, q != 0)))
    return p[tuple(idx)], q[tuple(idx)]

def kl(p, q):
    """Compute empirical KL divergence."""
    return np.sum(p * np.log(p / q))

def js(p, q):
    """Compute empirical JS divergence."""
    m = 0.5 * (p + q)
    return 0.5 * (kl(p, m) + kl(q, m))