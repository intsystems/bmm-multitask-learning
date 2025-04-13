""" Data generation utils
"""
from typing import Optional

import torch
from torch.utils.data import TensorDataset
from torch.distributions import Normal, Bernoulli


# TODO: may be change sharing params for linked and solo dataset later
SIGMA_X = 5.
SIGMA_FACTOR = 1.5


def build_linked_datasets(
    size: int,
    dim: int = 2
) -> tuple[TensorDataset]:
    EPS_X = 1e-2

    X = SIGMA_X * torch.randn((size, dim))
    X1 = X + EPS_X * torch.randn_like(X)
    X2 = X + EPS_X * torch.randn_like(X)
   
    EPS_W = 1.

    SIGMA_W = SIGMA_FACTOR * torch.ones((dim, ))
    SIGMA_W1 = SIGMA_W + EPS_W * (torch.rand_like(SIGMA_W) - 0.5)   # Uniform[-0.5, 0.5]
    SIGMA_W2 = SIGMA_W + EPS_W * (torch.rand_like(SIGMA_W) - 0.5)   # Uniform[-0.5, 0.5]
    w1 = Normal(X2.mean(dim=0), SIGMA_W1).sample()
    w2 = Normal(X1.mean(dim=0), SIGMA_W2).sample()

    y1 = Bernoulli(torch.sigmoid(X1.matmul(w1))).sample()
    y2 = Bernoulli(torch.sigmoid(X2.matmul(w2))).sample()

    return TensorDataset(X1, y1), TensorDataset(X2, y2)


def build_solo_dataset(
    size: int,
    dim: int = 2
) -> TensorDataset:
    X = SIGMA_X * torch.randn((size, dim))

    SIGMA_W = SIGMA_FACTOR * torch.ones((dim, ))
    # prior for w does not depend on any X
    w = Normal(torch.ones((dim, )), SIGMA_W).sample()

    y = Bernoulli(torch.sigmoid(X.matmul(w))).sample()

    return TensorDataset(X, y)
