""" Test correctness of elbo computation here
"""
import pytest

from itertools import chain

import torch
from torch import nn
from torch import distributions as distr

from bmm_multitask_learning.variational.elbo import MultiTaskElbo


def normal_suits():
    """ Latent ~ mean, cls ~ variance
    """
    TASK_NUM = 0
    # latents dim
    DIM = 2

    # Suite 1, similar distrs
    num_tasks = 3

    # somehow grad is not populated for distr params when distrs have the same numerical params
    scale_params = [nn.Parameter(torch.ones(DIM)) for _ in range(num_tasks)]
    cls_distrs = [
        distr.Independent(distr.Normal(torch.zeros_like(scale_param), scale_param), 1)
        for scale_param in scale_params
    ]

    # build random mixings
    mixing = torch.randn(num_tasks)
    mixing[0] = -torch.inf
    mixing = torch.softmax(mixing, dim=0)

    ans = torch.zeros(1)

    mt_elbo = MultiTaskElbo([None] * num_tasks, [None] * num_tasks, cls_distrs, ...)
    yield (mt_elbo, TASK_NUM, mixing, ans, [])

    # Suite 2
    num_tasks = 3

    loc_param = nn.Parameter(torch.ones(DIM))
    cls_distrs = [
        distr.Independent(distr.Normal(loc_param, torch.ones(DIM)), 1),
        distr.Independent(distr.Normal(torch.zeros(DIM), torch.ones(DIM)), 1),
        distr.Independent(distr.Normal(torch.zeros(DIM), torch.ones(DIM)), 1)
    ]
    
    mixing = torch.tensor([0., 0.5, 0.5])

    # from analytical formula
    ans = torch.ones(1)

    mt_elbo = MultiTaskElbo([None] * num_tasks, [None] * num_tasks, cls_distrs, ...)
    yield (mt_elbo, TASK_NUM, mixing, ans, [loc_param])


@pytest.mark.parametrize(
    "mt_elbo,task_num,cls_mixing,ans,must_have_grad",
    normal_suits()
)
def test_cls_kl_computation(
    mt_elbo: MultiTaskElbo,
    task_num: int,
    cls_mixing: torch.Tensor,
    ans: torch.Tensor,
    must_have_grad: list[torch.Tensor]
):
    """ 
    """
    assert torch.allclose(
        mt_elbo._compute_cls_kl_per_task(task_num, cls_mixing),
        ans
    )

    # check grads
    cls_mixing = cls_mixing.requires_grad_()
    lat_term = mt_elbo._compute_cls_kl_per_task(task_num, cls_mixing)
    lat_term.backward()
    assert cls_mixing.grad is not None
    for param in must_have_grad:
        assert param.grad is not None