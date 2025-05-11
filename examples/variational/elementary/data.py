""" Data generation utils
"""
import torch
from torch.utils.data import TensorDataset
from torch.distributions import Normal, Bernoulli


SIGMA_X = 5.
SIGMA_FACTOR = 1e-1

def build_linked_datasets(
    size: int,
    dim: int = 2
) -> tuple[TensorDataset]:
    EPS_X = 1e-2

    X = SIGMA_X * torch.randn((size, dim))
    X1 = X + EPS_X * torch.randn_like(X)
    X2 = X + EPS_X * torch.randn_like(X)

    SIGMA_W = SIGMA_FACTOR * torch.ones((dim, ))
    w1 = Normal(X2.mean(dim=0), SIGMA_W).sample()
    w2 = Normal(X1.mean(dim=0), SIGMA_W).sample()

    y1 = Bernoulli(logits=X1.matmul(w1)).sample()
    y2 = Bernoulli(logits=X2.matmul(w2)).sample()

    return (w1, TensorDataset(X1, y1)), (w2, TensorDataset(X2, y2))


def build_solo_dataset(
    size: int,
    dim: int = 2
) -> TensorDataset:
    X = SIGMA_X * torch.randn((size, dim))

    SIGMA_W = SIGMA_FACTOR * torch.ones((dim, ))
    # prior for w does not depend on any X
    w = Normal(torch.ones((dim, )), SIGMA_W).sample()

    y = Bernoulli(logits=X.matmul(w)).sample()

    return w, TensorDataset(X, y)
