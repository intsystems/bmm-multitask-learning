"""Utils for working with distributions
"""
from typing import Callable

import torch
from torch import distributions as distr


# conditional distribution for targets
type TargetDistr = Callable[[torch.Tensor, torch.Tensor], distr.Distribution]
# conditional distribution for latents
type LatentDistr = Callable[[torch.Tensor], distr.Distribution]
# conditional distribution for targets, but batched latents and classifiers
# must be flattened in one dimension. This is needed because of MixtureSameFamily design.
# See https://github.com/pytorch/pytorch/issues/76709 for future possible automation
type PredictiveDistr = Callable[[torch.Tensor, torch.Tensor], distr.Distribution]


def kl_sample_estimation(
    distr_1: distr.Distribution,
    distr_2: distr.Distribution,
    num_particles: int = 1
) -> torch.Tensor:
    samples = distr_1.rsample(num_particles)
    log_p_1 = distr_1.log_prob(samples)
    log_p_2 = distr_2.log_prob(samples)

    return (log_p_1 - log_p_2).mean()


def build_predictive(
    pred_distr: PredictiveDistr,
    classifier_distr: distr.Distribution,
    latent_distr: LatentDistr,
    X: torch.Tensor,
    classifier_num_particles: int = 1,
    latent_num_particles: int = 1
) -> distr.MixtureSameFamily:
    # sample hidden state (classifier + latent) from posterior
    classifier_samples = classifier_distr.sample((classifier_num_particles, ))
    latent_samples = latent_distr(X).sample((latent_num_particles, )).swapaxes(0, 1)
    # build conditional distribution objects for target
    pred_distr = pred_distr(latent_samples, classifier_samples)

    mixing_distr = distr.Categorical(torch.ones(pred_distr.batch_shape))

    return distr.MixtureSameFamily(mixing_distr, pred_distr)
