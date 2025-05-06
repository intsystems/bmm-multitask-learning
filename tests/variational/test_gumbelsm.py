import pytest

from itertools import chain, product

import torch

from bmm_multitask_learning.variational.elbo import MultiTaskElbo


MATRIX_SIZES = [2, 5, 10]

def random_mixing_matrices_iter():
    torch.manual_seed(42)
    NUM_MATRICES = 10
    SCALE = 5.

    for size in MATRIX_SIZES:
        for _ in range(NUM_MATRICES):
            yield SCALE * torch.randn(size, size).abs()

def random_temps_iter():
    torch.manual_seed(42)
    NUM_TEMPS = 10
    SCALE = 1.5

    for _ in range(NUM_TEMPS):
        yield SCALE * torch.randn(1).abs()

def random_uniform_mixing_matrices_iter():
    torch.manual_seed(42)
    NUM_MATRICES = 10
    SCALE = 5.

    for size in MATRIX_SIZES:
        for _ in range(NUM_MATRICES):
            yield SCALE * torch.randn(size).abs()[:, None].expand((-1, size))

def make_mixing_matrix(mixing_params: torch.Tensor, t: float):
    # make dummy mt_elbo class
    num_tasks = mixing_params.shape[0]
    mt_elbo = MultiTaskElbo([None] * num_tasks, ..., ..., ...)
    return mt_elbo._get_gumbelsm_mixing(mixing_params, t)


@pytest.mark.parametrize(
    "mixing_params,t",
    product(chain(random_mixing_matrices_iter(), random_uniform_mixing_matrices_iter()), random_temps_iter())
)
def test_mixing_matrix_correctness(mixing_params: torch.Tensor, t: float):
    mixing_matrix = make_mixing_matrix(mixing_params, t)

    # matrix must be square
    assert mixing_matrix.ndim == 2
    assert mixing_matrix.shape[0] == mixing_matrix.shape[1]

    assert torch.all(mixing_matrix <= 1.)
    assert torch.all(mixing_matrix >= 0.)

    mixing_diag = mixing_matrix.diag()
    assert torch.allclose(mixing_diag, torch.zeros_like(mixing_diag))

    row_sums = mixing_matrix.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))


@pytest.mark.parametrize(
    "mixing_params",
    random_mixing_matrices_iter()
)
def test_mixing_matrix_is_categorical(mixing_params: torch.Tensor):
    num_tasks = mixing_params.shape[0]
    # make temperature very low
    t: float = 1e-10
    mixing_matrix = make_mixing_matrix(mixing_params, t)

    # each row must contain one 1. and rest are 0.
    assert torch.allclose(
        torch.isclose(mixing_matrix, torch.ones_like(mixing_matrix)).float().sum(dim=1),
        torch.ones(num_tasks)
    )
    assert torch.allclose(
        torch.isclose(mixing_matrix, torch.zeros_like(mixing_matrix)).float().sum(dim=1),
        (num_tasks - 1) * torch.ones(num_tasks)
    )


@pytest.mark.parametrize(
    "mixing_params",
    random_mixing_matrices_iter()
)
def test_mixing_matrix_is_uniform(mixing_params: torch.Tensor):
    num_tasks = mixing_params.shape[0]
    # make temperature very high
    t: float = 1e20
    mixing_matrix = make_mixing_matrix(mixing_params, t)

    assert torch.equal(
        torch.isclose(
            mixing_matrix, (1 / (num_tasks - 1)) * torch.ones_like(mixing_matrix)
        ).int().sum(dim=1),
        (num_tasks - 1) * torch.ones(num_tasks).int()
    )
