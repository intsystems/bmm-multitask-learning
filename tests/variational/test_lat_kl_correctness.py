""" Test correctness of elbo computation here
"""
import pytest

from pipe import select
from itertools import chain

import torch
from torch import nn
from torch import distributions as distr

from bmm_multitask_learning.variational.elbo import MultiTaskElbo


def normal_suits():
    """ 
    """
    TASK_NUM = 0
    # latents dim
    DIM = 2

    # Suite 1, similar distrs
    num_tasks = 3
    num_samples = [1, 5, 50, 100]

    inputs = [torch.randn(i, DIM) for i in num_samples]
    # somehow grad is not populated for distr params when distrs have the same numerical params
    scale_params = [nn.Parameter(torch.ones(DIM)) for _ in range(num_tasks)]
    lat_distrs = [
        lambda inputs: distr.Independent(distr.Normal(torch.zeros_like(inputs), scale_param.expand_as(inputs)), 1)
        for scale_param in scale_params
    ]
    mixings = []
    # build random mixings
    for i in range(len(num_samples)):
        mixing = torch.randn(num_tasks)
        mixing[0] = -torch.inf
        mixing = torch.softmax(mixing, dim=0)
        mixings.append(mixing)
    ans = [torch.zeros(1) for _ in num_samples]

    for i, n_s in enumerate(num_samples):
        mt_elbo = MultiTaskElbo([None] * num_tasks, [n_s] * num_tasks, ..., lat_distrs)
        yield (mt_elbo, TASK_NUM, mixings[i], inputs[i], ans[i], [])

    # Suite 2
    num_tasks = 3
    num_samples = [1, 5, 50]

    scale_param = nn.Parameter(torch.ones(DIM))
    lat_distrs = [
        lambda inputs: distr.Independent(distr.Normal(inputs, scale_param.expand_as(inputs)), 1),
        lambda inputs: distr.Independent(distr.Normal(torch.zeros_like(inputs), torch.ones_like(inputs)), 1),
        lambda inputs: distr.Independent(distr.Normal(torch.zeros_like(inputs), torch.ones_like(inputs)), 1)
    ]
    inputs = [torch.randn(i, DIM) for i in num_samples]
    mixing = torch.tensor([0., 0.5, 0.5])
    # from analytical formula
    ans = [0.5 * (inp ** 2).sum(dim=1).sum(dim=0) for inp in inputs]

    for i, n_s in enumerate(num_samples):
        mt_elbo = MultiTaskElbo([None] * num_tasks, [n_s] * num_tasks, ..., lat_distrs)
        yield (mt_elbo, TASK_NUM, mixing, inputs[i], ans[i], [scale_param])


@pytest.mark.parametrize(
    "mt_elbo,task_num,lat_mixing,inputs,ans,must_have_grad",
    normal_suits()
)
def test_lat_kl_computation(
    mt_elbo: MultiTaskElbo,
    task_num: int,
    lat_mixing: torch.Tensor,
    inputs: torch.Tensor,
    ans: torch.Tensor,
    must_have_grad: list[torch.Tensor]
):
    """ 
    """
    assert torch.allclose(
        mt_elbo._compute_latent_kl_per_task(task_num, inputs, lat_mixing),
        ans
    )

    # check grads
    lat_mixing = lat_mixing.requires_grad_()
    lat_term = mt_elbo._compute_latent_kl_per_task(task_num, inputs, lat_mixing)
    lat_term.backward()
    assert lat_mixing.grad is not None
    for param in must_have_grad:
        assert param.grad is not None
