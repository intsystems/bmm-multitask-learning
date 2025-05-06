import torch
import torch.nn as nn
import pytest
from bmm_multitask_learning.sbmtl.sparse_bayesian_regression import SparseBayesianRegression

def test_init_hyperparams():
    model = nn.Linear(10, 5, bias=False)
    group_indices = [list(range(2)), list(range(2, 4)), list(range(4, 6)), list(range(6, 8)), list(range(8, 10))]
    sbr = SparseBayesianRegression(model, group_indices, device='cpu')
    assert sbr.M_W.shape == (5, 10)
    assert len(sbr.M_Z) == len(group_indices)
    assert sbr.M_V.shape == (5, 2)

def test_fit_and_predict():
    n_samples = 200
    n_features = 10
    n_tasks = 5
    X = torch.randn(n_samples, n_features)
    true_w = torch.zeros(n_tasks, n_features)
    true_w[:, :3] = torch.randn(n_tasks, 3)
    Y = true_w @ X.T + 0.5 * torch.randn(n_tasks, n_samples)

    model = nn.Linear(n_features, n_tasks, bias=False)
    group_indices = [list(range(2)), list(range(2, 4)), list(range(4, 6)), list(range(6, 8)), list(range(8, 10))]
    sbr = SparseBayesianRegression(model, group_indices, device='cpu')
    sbr.fit(X, Y, num_iter=10)

    y_pred = sbr.predict(X)
    assert y_pred.shape == Y.shape, "Размерность должна быть (число_задач, число_объектов)"

def test_m_step():
    model = nn.Linear(10, 5, bias=False)
    group_indices = [list(range(2)), list(range(2, 4)), list(range(4, 6)), list(range(6, 8)), list(range(8, 10))]
    sbr = SparseBayesianRegression(model, group_indices, device='cpu')
    post = {
        'mean_gammas': [1.0] * len(group_indices),
        'mean_inv_gammas': [1.0] * len(group_indices),
        'mean_log_gammas': [1.0] * len(group_indices)
    }
    sbr.m_step(post)
    assert sbr.omega_prior is not None
    assert sbr.chi_prior is not None
    assert sbr.phi_prior is not None

def test_e_step():
    n_samples = 200
    n_features = 10
    n_tasks = 5
    X = torch.randn(n_samples, n_features).T
    Y = torch.randn(n_tasks, n_samples)

    model = nn.Linear(n_features, n_tasks, bias=False)
    group_indices = [list(range(2)), list(range(2, 4)), list(range(4, 6)), list(range(6, 8)), list(range(8, 10))]
    sbr = SparseBayesianRegression(model, group_indices, device='cpu')
    post = sbr.e_step(X, Y)
    assert 'mean_gammas' in post
    assert 'mean_inv_gammas' in post
    assert 'mean_log_gammas' in post
    assert post['M_W'].shape == (n_tasks, n_features)