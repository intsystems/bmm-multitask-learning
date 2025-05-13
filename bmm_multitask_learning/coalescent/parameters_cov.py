import numpy as np

from .coalescent_inference import CoalescentTree


def optimal_cov(coalesce_tree: CoalescentTree, dim):
    """
    retunrns samples from covariance matrix
    for InverseWishart distribution parameters update
    """
    sample_mean = np.zeros((dim, ), dtype=float)
    overall_items = 0

    # compute sample mean
    for item in coalesce_tree.iterate_over():
        parent = item.parent
        if parent is not None:
            overall_items += 1
            dt = item.t - parent.t
            D_i = item.mean - parent.mean
            x_i = D_i / np.sqrt(dt)
            sample_mean += x_i
    sample_mean /= overall_items

    
    # compute samples
    samples = []

    for item in coalesce_tree.iterate_over():
        parent = item.parent
        if parent is not None:
            dt = item.t - parent.t
            D_i = item.mean - parent.mean
            x_i = D_i / np.sqrt(dt)
            samples.append(x_i - sample_mean)

    return np.array(samples)


def optimal_R(s_leaves, weights):
    y = np.exp(- s_leaves) * weights  # n x d
    return y
