from typing import Literal, Callable
from functools import partial
from statistics import mean
from pipe import select
from itertools import product

import torch
from torch import nn
from torch import distributions as distr

from .distr import kl_sample_estimation, TargetDistr, LatentDistr


class MultiTaskElbo(nn.Module):
    """General ELBO computer for variational multitask problem. 
    """
    def __init__(
        self,
        task_distrs: list[TargetDistr],
        task_num_samples: list[int],
        classifier_distr: list[distr.Distribution],
        latent_distr: list[LatentDistr],
        classifier_num_particles: int = 1,
        latent_num_particles: int = 1,
        temp_scheduler: Callable | Literal["const"] = Literal["const"],
        kl_estimator_num_samples: int = 10
    ):
        """
        TODO: rewrite docstring
        Args:
            task_distrs (list[distr.Distribution]): Data distribution for each task p_t(y | z, w)
            task_num_samples (list[int]): Number of train samples for each task. Needed for unbiased ELBO computation in case of batched data.
            classifier_distr (distr.Distribution): Distribution for the classifier q(w | D)
            latent_distr (distr.Distribution): Distribution for the latent state q(z | x, D)
            classifier_num_particles (int, optional): num samples from classifier distr. Defaults to 1.
            latent_num_particles (int, optional): num samples from latent distr. Defaults to 1.
            temp_scheduler (Callable | Literal[&quot;const&quot;], optional): _description_. Defaults to Literal["const"].
        """
        super().__init__()

        self.task_distrs = task_distrs
        self.classifier_distr = classifier_distr
        self.latent_distr = latent_distr

        self.num_tasks = len(task_distrs)
        if self.num_tasks < 2:
            raise ValueError(f"Number of tasks should be > 2, {self.num_tasks} was given")
        self.task_num_samples = task_num_samples
        self.classifier_num_particles = classifier_num_particles
        self.latent_num_particles = latent_num_particles
        self.kl_estimator_num_samples = kl_estimator_num_samples

        self.temp_scheduler = temp_scheduler if temp_scheduler is not "const" else lambda t: 1.

        # define gumbel-softmax parameters for classifier and latent
        # initialize uniform
        self._classifier_mixings_params, self._latent_mixings_params = [
            nn.Parameter(
                torch.full((self.num_tasks, self.num_tasks), 1 / (self.num_tasks - 1))
            )
        ] * 2

    def forward(self, targets: list[torch.Tensor], data: list[torch.Tensor], step: int) -> torch.Tensor:
        """Computes ELBO estimation for variational multitask problem.

        Args:
            targets (list[torch.Tensor]): batched targets (y) for each task 
            data (list[torch.Tensor]): batched data (X) for each task 
            step: needed for temperature func

        Returns:
            torch.Tensor: ELBO estimation
        """
        batch_sizes = list(data | select(lambda x: x.shape[0]))

        # get mixing values in form of matrix
        temp = self.temp_scheduler(step)
        classifier_mixing = self._get_gumbelsm_mixing(self._classifier_mixings_params, temp)
        latent_mixing = self._get_gumbelsm_mixing(self._latent_mixings_params, temp)

        # sample classifiers
        # shape = (num_tasks, classifier_num_particles, classifier_shape)
        classifiers = torch.stack(
            list(
                self.classifier_distr |
                select(lambda d: d.rsample((self.classifier_num_particles, )))
            )
        )

        # sample latents
        # shape = (num_tasks, num_samples(num_tasks), latent_num_particles, latent_shape)
        latents = []
        for i, latent_cond_distr in enumerate(self.latent_distr):
            latent_samples_per_data = []
            for sample in data[i]:
                latent_distr = latent_cond_distr(sample)
                latent_samples_per_data.append(latent_distr.rsample((self.latent_num_particles, )))
            latents.append(latent_samples_per_data)

        # get log liklyhood for task + sampled averaged across latent and classifier particles
        lh_per_task = []
        for i, task_cond_distr in enumerate(self.task_distrs):
            sample_lh = []
            for j, sample in enumerate(targets[i]):
                sample_lh.append(
                    # try use "mean" function
                    # sum(
                    #     product(latents[i][j], classifiers[i]) |
                    #     select(lambda lat_cl: task_cond_distr(*lat_cl).log_prob(sample))
                    # ) / (self.latent_num_particles * self.classifier_num_particles)
                    mean(
                        product(latents[i][j], classifiers[i]) |
                        select(lambda lat_cl: task_cond_distr(*lat_cl).log_prob(sample))
                    )
                )
            lh_per_task.append(sample_lh)
        # sum lh samples for each task (regarding butch size correction)
        # and average across tasks
        lh_val = []
        for i, sample_lh in enumerate(lh_per_task):
            lh_val.append((self.num_tasks[i] / batch_sizes[i]) * sum(sample_lh))
        lh_val = mean(lh_per_task)

        # get latents kl for each task, for each datum in task
        # shape = (num_tasks, num_samples(num_tasks))
        latents_kl = []
        for i in range(len(self.latent_distr)):
            kl_for_samples = []
            for sample in data[i]:
                latent_distrs = [latent_cond_distr(sample) for latent_cond_distr in self.latent_distr]
                # fix distribution for task i in kl function as first argument
                kl_computer = partial(self._compute_kl, distr_1=latent_distrs[i])
                individual_kls = torch.tensor([kl_computer(latent_distr) for latent_distr in latent_distr])
                # this is an estimation of the kl for mixed distribution
                full_kl = individual_kls.dot(latent_mixing[i])
                kl_for_samples.append(full_kl)
            latents_kl.append(kl_for_samples)
        # sum latents kl for each task and average among tasks
        latents_kl: torch.Tensor = sum(latents_kl | select(sum)) / self.num_tasks
        
        # get classifiers kl for each task
        classifiers_kl = []
        for i, classifier_distr in enumerate(self.classifier_distr):
            kl_computer = partial(self._compute_kl, distr_1=classifier_distr)
            individual_kls = torch.tensor([kl_computer(other_cl_distr) for other_cl_distr in self.classifier_distr])
            full_kl = individual_kls.dot(classifier_mixing[i])
            classifiers_kl.append(full_kl)
        # average classifiers kl
        classifiers_kl = sum(classifiers_kl) / self.num_tasks

        return lh_val + latents_kl + classifiers_kl

    def _compute_kl(self, distr_1: distr.Distribution, distr_2: distr.Distribution) -> torch.Tensor:
        """Computes KL analytically if possible else make a sample estimation
        """
        if distr_1 is distr_2:
            return 0.

        try:
            return distr.kl_divergence(distr_1, distr_2)
        except NotImplementedError:
            return kl_sample_estimation(distr_1, distr_2, self.kl_estimator_num_samples)

    def _get_gumbelsm_mixing(self, mixings_params: torch.Tensor, temp: float) -> torch.Tensor:
        # mixing with self is prohibited, so we mask diagonal to get zeros after softmax
        mask = torch.eye(self.num_tasks) * float("-inf")

        mixing = distr.Gumbel(0., 1.).sample((self.num_tasks, self.num_tasks))
        mixing += mixings_params.log()
        mixing = mixing / temp
        mixing += mask
        mixing = torch.softmax(mixing, dim=1)

        return mixing

    @property
    def classifier_mixings_params(self):
        return self._classifier_mixings_params

    @property
    def latent_mixings_params(self):
        return self._latent_mixings_params
