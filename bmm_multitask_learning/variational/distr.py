"""Utils for working with distributions
"""
from typing import Callable
from itertools import product
from pipe import select

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


class Predictive(distr.Distribution):
    def __init__(
        self,
        task_distr: TargetDistr,
        classifier_distr: distr.Distribution,
        latent_distr: LatentDistr,
        x: torch.Tensor,
        classifier_num_particles: int = 1,
        latent_num_particles: int = 1,
    ):
        # sample hidden state (classifier + latent) from posterior
        classifier_samples = classifier_distr.sample((classifier_num_particles, ))
        latent_samples = latent_distr(x).sample((latent_num_particles, ))
        # build conditional distribution objects for target
        self.target_distrs = list(
            product(classifier_samples, latent_samples) |
            select(lambda cl_lt: task_distr(*cl_lt))
        )

        target_distr_example: distr.Distribution = self.target_distrs[0]
        super().__init__(
            batch_shape=target_distr_example.batch_shape, 
            event_shape=target_distr_example.event_shape, 
            validate_args=target_distr_example._validate_args
        )

    def log_prob(self, value):
        return torch.concat(list(
            self.target_distrs | select(lambda distr: distr.log_prob(value))
        )).mean()

    def sample(self, sample_shape = ...):
        # choose distr from mix
        indx = torch.randint(0, len(self.target_distrs), sample_shape).flatten().tolist()
        # sample from chosen distr
        samples = torch.concat(list(indx | select(lambda i: self.target_distrs[i].sample())))
        # tune shape for the output samples
        samples = samples.reshape(*sample_shape, -1)

        return samples
