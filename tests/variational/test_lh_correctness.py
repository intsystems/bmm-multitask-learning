""" Test correctness of elbo computation here
"""
import pytest

from itertools import chain, repeat

import torch
from torch import distributions as distr

from bmm_multitask_learning.variational.elbo import MultiTaskElbo


def normal_suits():
    """ Latent ~ mean, cls ~ variance
    """
    torch.manual_seed(42)

    # minimum num of tasks
    NUM_TASKS = 2
    # targets dim
    DIM = 2
    # diagonal normals
    def task_distr(loc: torch.Tensor, scale: torch.Tensor):
        scale_particles = scale.shape[0]
        new_shape = list(loc.shape)
        new_shape.insert(2, scale_particles)

        loc = loc[:, :, None, ...].expand(*new_shape)
        scale = scale[None, None, ...].expand(*new_shape)

        return distr.Independent(distr.Normal(loc, scale), 1)

    task_distrs = [task_distr for _ in range(NUM_TASKS)]

    # Suite 1
    num_samples = 1
    num_particles = 1
    latents = torch.zeros((num_samples, num_particles, DIM))
    classifiers = torch.ones((num_particles, DIM))
    targets = torch.zeros((num_samples, DIM))
    ans = -num_samples * torch.tensor(1 / (2 * torch.pi) ** (DIM / 2)).log()
    mt_elbo = MultiTaskElbo(task_distrs, [num_samples] * NUM_TASKS, ..., ..., num_particles, num_particles)
    yield (mt_elbo, latents, classifiers, targets, ans)

    # Suite 2, add dummy particles and samples
    num_samples = 100
    num_particles = 100
    latents = torch.zeros((num_samples, num_particles, DIM))
    classifiers = torch.ones((num_particles, DIM))
    targets = torch.zeros((num_samples, DIM))
    ans = -num_samples * torch.tensor(1 / (2 * torch.pi) ** (DIM / 2)).log()
    mt_elbo = MultiTaskElbo(task_distrs, [num_samples] * NUM_TASKS, ..., ..., num_particles, num_particles)
    yield (mt_elbo, latents, classifiers, targets, ans)

    # Suite 3, change mean and targets
    num_samples = 100
    num_particles = 100
    latents = 10 * torch.randn((num_samples, DIM))[:, None, :].expand(-1, num_particles, -1)
    classifiers = torch.ones((num_particles, DIM))
    targets = latents[:, 0, :].clone()
    ans = -num_samples * torch.tensor(1 / (2 * torch.pi) ** (DIM / 2)).log()
    mt_elbo = MultiTaskElbo(task_distrs, [num_samples] * NUM_TASKS, ..., ..., num_particles, num_particles)
    yield (mt_elbo, latents, classifiers, targets, ans)

def bernouli_suits():
    torch.manual_seed(42)

    # minimum num of tasks
    NUM_TASKS = 2
    # targets dim
    DIM = 2
    task_distr = lambda lat, cls: distr.Bernoulli(logits=(lat*cls).sum(-1))
    task_distrs = [task_distr for _ in range(NUM_TASKS)]

    # Suite 1
    num_samples = 1
    num_particles = 1
    latents = torch.ones((num_samples, num_particles, DIM))
    classifiers = torch.zeros((num_particles, DIM))
    targets = torch.zeros((num_samples, ))
    ans = -torch.tensor(0.5).log() * num_samples
    mt_elbo = MultiTaskElbo(task_distrs, [num_samples] * NUM_TASKS, ..., ..., num_particles, num_particles)
    yield (mt_elbo, latents, classifiers, targets, ans)

    # Suite 2
    num_samples = 10
    num_particles = 10
    latents = torch.ones((num_samples, num_particles, DIM))
    classifiers = torch.zeros((num_particles, DIM))
    targets = torch.zeros((num_samples, ))
    ans = -torch.tensor(0.5).log() * num_samples
    mt_elbo = MultiTaskElbo(task_distrs, [num_samples] * NUM_TASKS, ..., ..., num_particles, num_particles)
    yield (mt_elbo, latents, classifiers, targets, ans)

    # Suite 3
    num_samples = 10
    num_particles = 10
    latents = torch.zeros((num_samples, num_particles, DIM))
    classifiers = torch.ones((num_particles, DIM))
    targets = torch.zeros((num_samples, ))
    ans = -torch.tensor(0.5).log() * num_samples
    mt_elbo = MultiTaskElbo(task_distrs, [num_samples] * NUM_TASKS, ..., ..., num_particles, num_particles)
    yield (mt_elbo, latents, classifiers, targets, ans)

def uniform_suits():
    torch.manual_seed(42)

    # minimum num of tasks
    NUM_TASKS = 2
    # targets dim
    DIM = 2
    task_distr = lambda lat, cls: distr.Independent(distr.Uniform(lat, cls), 1)
    task_distrs = [task_distr for _ in range(NUM_TASKS)]

    # Suite 1
    num_samples = 1
    num_particles = 1
    latents = torch.zeros((num_samples, num_particles, DIM))
    classifiers = torch.ones((num_particles, DIM))
    targets = torch.rand((num_samples, DIM))
    ans = -torch.tensor(1.).log() * num_samples
    mt_elbo = MultiTaskElbo(task_distrs, [num_samples] * NUM_TASKS, ..., ..., num_particles, num_particles)
    yield (mt_elbo, latents, classifiers, targets, ans)

    # Suite 2
    num_samples = 100
    num_particles = 100
    latents = torch.zeros((num_samples, num_particles, DIM))
    classifiers = torch.ones((num_particles, DIM))
    targets = torch.rand((num_samples, DIM))
    ans = -torch.tensor(1.).log() * num_samples
    mt_elbo = MultiTaskElbo(task_distrs, [num_samples] * NUM_TASKS, ..., ..., num_particles, num_particles)
    yield (mt_elbo, latents, classifiers, targets, ans)


@pytest.mark.parametrize(
    "mt_elbo,latents,classifiers,targets,ans",
    chain(normal_suits(), bernouli_suits(), uniform_suits())
)
def test_lh_computation(
    mt_elbo: MultiTaskElbo,
    latents: torch.Tensor,
    classifiers: torch.Tensor,
    targets: torch.Tensor,
    ans: torch.Tensor
):
    """ Test correctness of lh computation for simple distributions
    """
    # task num does not matter
    TASK_NUM = 0
    assert torch.allclose(
        mt_elbo._compute_lh_per_task(TASK_NUM, latents, classifiers, targets),
        ans
    )

    # check grad operations
    latents = latents.requires_grad_()
    classifiers = classifiers.requires_grad_()
    lh =  mt_elbo._compute_lh_per_task(TASK_NUM, latents, classifiers, targets)
    lh.backward()
    assert latents.grad is not None
    assert classifiers.grad is not None


