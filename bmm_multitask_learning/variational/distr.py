"""Utils for working with distributions
"""
from typing import Callable

import torch
from torch import nn
from torch import distributions as distr


# conditional distribution for targets
type TargetDistr = Callable[[torch.Tensor, torch.Tensor], distr.Distribution]
# conditional distribution for latents
type LatentDistr = Callable[[torch.Tensor], distr.Distribution]


def kl_sample_estimation(
    distr_1: distr.Distribution,
    distr_2: distr.Distribution,
    num_particles: int = 1
) -> torch.Tensor:
    samples = distr_1.rsample(num_particles)
    log_p_1 = distr_1.log_prob(samples)
    log_p_2 = distr_2.log_prob(samples)

    return (log_p_1 - log_p_2).mean()


class Predictive():
    ...