#!/usr/bin/env python
# coding: utf-8

# # Variational multitask learning elementary example

# This is a practical demonstration on how to use `variational` subpackage on a simple classification example. We are going to solve 3 classification tasks with logistic regression as a model. Additionally, we will add prior on the weight so the tasks become bayessian. Two of the tasks will be probabilsitcally connected, the last will have no probabilistic connections with others.
# 
# First, we will apply variational principle to learn each task individually. We will use [`pyro`](https://pyro.ai/examples/index.html) package to automatically compute ELBO and minimize it.
# 
# Secondly, we will use `variational` subpackage and learn 3 tasks alltogether. Learning here is the same ELBO minimizing, but for special variational structure - see [doc](../../../docs/variational/intro.md) for more details.
# 
# Lastly, we will compare two approaches in accuracy terms.

# In[ ]:


from pipe import select

from omegaconf import OmegaConf

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from pyro.distributions import Delta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, StackDataset
from torch import optim
from torch import distributions as distr

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping

from torchmetrics.classification import Accuracy

from bmm_multitask_learning.variational.elbo import MultiTaskElbo
from bmm_multitask_learning.variational.distr import build_predictive
from data import build_linked_datasets, build_solo_dataset


def test_on_elemntary_example():
    config = OmegaConf.load("tests/variational/elementary_test/config.yaml")
    
    torch.manual_seed(config.seed);
    
    NUM_MODELS = 3
    
    datasets = [*build_linked_datasets(config.size, config.dim), build_solo_dataset(config.size, config.dim)]
    # extract w
    w_list = list(
        datasets | select(lambda w_dataset: w_dataset[0])
    )
    # extract (X, y) pairs
    datasets = list(
        datasets |
        select(lambda w_dataset: w_dataset[1]) |
        select(lambda d: random_split(d, [1 - config.test_ratio, config.test_ratio]))
    )
    
    train_datasets, test_datasets = zip(*datasets)

    points_df = []
    
    for i, dataset in enumerate(train_datasets):
        X = dataset.dataset.tensors[0].numpy()
        cur_df = pd.DataFrame(X, columns=["x1", "x2"])
        cur_df["dataset"] = str(i)
        points_df.append(cur_df)
    points_df = pd.concat(points_df, axis=0)
    
    points_df.head()
    
    @distr.kl.register_kl(Delta, Delta)
    def kl_delta_delta(d1: Delta, d2: Delta):
        return torch.zeros(d1.batch_shape)
        # this is how it should be
        # return torch.zeros(d1.batch_shape) if torch.allclose(d1.v, d2.v) else torch.full(torch.inf, d1.batch_shape)
    
    
    # Define batched distributions for tasks. Here we assume that *latents* come with (batch_size, num_latent_particles, ...) shape, *classifiers* come with (num_classifier_samples, ...) shape
    
    # same for all tasks
    def target_distr(Z: torch.Tensor, W: torch.Tensor) -> distr.Distribution:
        return distr.Bernoulli(logits=torch.tensordot(Z, W, dims=[[-1], [-1]]))
    
    # same for all tasks
    def predictive_distr(Z: torch.Tensor, W: torch.Tensor) -> distr.Distribution:
        return distr.Bernoulli(logits=torch.tensordot(Z, W, dims=[[-1], [-1]]).flatten(1, 2))

    task_distrs = [target_distr for _ in range(NUM_MODELS)]
    task_num_samples = list(train_datasets | select(len))

    # we don't have latents here, but we need it formaly as distribution
    def latent_distr(X: torch.Tensor) -> distr.Distribution:
        return Delta(X, event_dim=1)
    
    class NormalLogits(distr.Normal):
        """Normal distribution with scale parametrized via logits
        """
        def __init__(self, loc, logit, validate_args=None):
            self.logit = logit
            super().__init__(loc, torch.exp(logit), validate_args)
    
        def __getattribute__(self, name):
            if name == "scale":
                return self.logit.exp()
            else:
                return super().__getattribute__(name)
    
    
    # parametric variational distr for classifiers
    classifier_distrs_params = {}
    classifier_distrs = []
    for i in range(NUM_MODELS):
        # set inital values for distribution's parameters
        loc, scale_logit = nn.Parameter(torch.zeros((config.dim, ))), nn.Parameter(torch.zeros((config.dim, )))
        classifier_distrs_params.update({
            f"distr_{i}": [loc, scale_logit]
        })
        classifier_distrs.append(
            distr.Independent(
                NormalLogits(loc, scale_logit),
                reinterpreted_batch_ndims=1
            )
        )
    # parametric variational distr for latents
    latent_distrs = [latent_distr for _ in range(NUM_MODELS)]

    # temperature must decrease over steps
    temp_scheduler = lambda step: 1. / torch.sqrt(torch.tensor(step + 1))
    
    # create variational multitask elbo module
    mt_elbo = MultiTaskElbo(
        task_distrs,
        task_num_samples,
        classifier_distrs,
        latent_distrs,
        temp_scheduler=temp_scheduler,
        **dict(config.mt_elbo)
    )
    
    class LitMtModel(L.LightningModule):
        def __init__(
            self,
            mt_elbo: MultiTaskElbo
        ):
            super().__init__()
    
            num_tasks = mt_elbo.num_tasks
            self.accuracy_computers = [Accuracy('binary') for _ in range(num_tasks)]
            self.mt_elbo = mt_elbo
    
            self.distr_params = nn.ParameterList()
            for param_list in classifier_distrs_params.values():
                self.distr_params.extend(
                    param_list
                )
    
    
        def training_step(self, batch: tuple[tuple[torch.Tensor]], batch_idx: int):
            mt_loss_dict = self.mt_elbo(*list(zip(*batch)), step=self.global_step)
            self.log_dict(mt_loss_dict, prog_bar=True)
    
            # DEBUG
            return mt_loss_dict["elbo"]
    
        def on_train_batch_end(self, outputs, batch, batch_idx):
            with torch.no_grad():
                for distr_name, distr_params in classifier_distrs_params.items():
                    params_grad_norm = sum(distr_params | select(lambda param: param.grad.norm()))
                    self.log(f"{distr_name}_grad", params_grad_norm)
    
        def validation_step(self, batch: tuple[tuple[torch.Tensor]], batch_idx: int):
            for i, (X, y) in enumerate(batch):
                NUM_PREDICTIVE_SAMPLES = 10
                cur_predictive = build_predictive(
                    predictive_distr,
                    classifier_distrs[i],
                    latent_distrs[i],
                    X,
                    config.mt_elbo.classifier_num_particles,
                    config.mt_elbo.latent_num_particles
                )
                y_pred = (cur_predictive.sample((NUM_PREDICTIVE_SAMPLES, )).mean(dim=0) > 0.5).float()
                self.accuracy_computers[i].update(y_pred, y)
    
        def on_validation_epoch_end(self):
            for i, accuracy_computer in enumerate(self.accuracy_computers):
                self.log(f"Test/Accuracy_{i}", accuracy_computer.compute())
                accuracy_computer.reset()
    
        def configure_optimizers(self):
            return optim.Adam(self.parameters(), lr=1e-3)
    
    lit_mt_model = LitMtModel(mt_elbo)
    
    # stack task datasets
    unified_train_dataset = StackDataset(*train_datasets)
    unified_test_dataset = StackDataset(*test_datasets)
    
    mt_train_dataloader = DataLoader(unified_train_dataset, config.batch_size, shuffle=True)
    mt_test_dataloader = DataLoader(unified_test_dataset, config.batch_size, shuffle=False)
    
    callbacks = [
        EarlyStopping(monitor="elbo", min_delta=1e-3, patience=10, mode="min")
    ]
    
    trainer = L.Trainer(callbacks=callbacks, enable_checkpointing=False, logger=False, **dict(config.trainer))
    trainer.fit(lit_mt_model, mt_train_dataloader, mt_test_dataloader)