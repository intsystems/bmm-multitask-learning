"""Utils for working with distributions
"""
from typing import Literal, Callable

import torch
from torch import nn
from torch import distributions as distr


def kl_sample_estimation(
    distr_1: distr.Distribution,
    distr_2: distr.Distribution,
    num_particles: int = 1
) -> torch.Tensor:
    samples = distr_1.rsample(num_particles)
    p_1 = distr_1.log_prob(samples).exp()
    p_2 = distr_2.log_prob(samples).exp()

    return (p_1 * torch.log(p_1 / p_2)).sum()


class Predictive():
    ...