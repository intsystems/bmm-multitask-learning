import pytest

import torch
from torch.distributions import Normal, kl_divergence

from bmm_multitask_learning.variational.distr import kl_sample_estimation


def test_kl_esimation():
    NUM_PARTICLES = 100

    distr1 = Normal(torch.zeros(2), torch.ones(2))
    distr2 = Normal(torch.zeros(2), torch.ones(2))

    assert torch.allclose(
        kl_divergence(distr1, distr2),
        kl_sample_estimation(distr1, distr2, NUM_PARTICLES)
    )
