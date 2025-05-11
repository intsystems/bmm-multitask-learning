from typing import Literal, Callable
from functools import partial
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
        temp_scheduler: Callable[[int], float] | Literal["const"] = Literal["const"],
        kl_estimator_num_samples: int = 10
    ):
        """
        Args:
            task_distrs (list[TargetDistr]): Data distribution for each task p_t(y | z, w)
            task_num_samples (list[int]): Number of train samples for each task. Needed for unbiased ELBO computation in case of batched data.
            classifier_distr (list[distr.Distribution]): Distribution for the classifier q(w | D)
            latent_distr (list[LatentDistr]): Distribution for the latent state q(z | x, D)
            classifier_num_particles (int, optional): num samples from classifier distr. Defaults to 1.
            latent_num_particles (int, optional):  num samples from latent distr. Defaults to 1.
            temp_scheduler (Callable[[int], float] | Literal[&quot;const&quot;], optional): _description_. Defaults to Literal["const"].
            kl_estimator_num_samples (int, optional): if your distrs does not have implicit kl computation, 
            it will be approximated using this number of samples. Defaults to 10.

            Warning:
                This nn.Module does not register nn.Parameters from the distributions inside itself
        Raises:
            ValueError: if number of tasks <= 2
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

    def forward(self, data: list[torch.Tensor], targets: list[torch.Tensor], step: int) -> torch.Tensor:
        """Computes ELBO estimation for variational multitask problem.

        Args:
            targets (list[torch.Tensor]): batched targets (y) for each task 
            data (list[torch.Tensor]): batched data (X) for each task 
            step: needed for temperature func

        Returns:
            torch.Tensor: ELBO estimation
        """
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
        # shape = [num_tasks, (num_samples(num_tasks), latent_num_particles, latent_shape)]
        latents = []
        for i, latent_cond_distr in enumerate(self.latent_distr):
            latents.append(
                latent_cond_distr(data[i]).rsample((self.latent_num_particles, )).swapaxes(0, 1)
            )

        # get log liklyhood for task + sampled averaged across latent and classifier particles
        lh_per_task = []
        for i in range(self.num_tasks):
            cur_lh = self._compute_lh_per_task(i, latents[i], classifiers[i], targets[i])
            lh_per_task.append(cur_lh)
        # average lh samples across tasks
        lh_val = torch.stack(lh_per_task).mean()

        # get summed latents kl for each task
        latents_kl = []
        for i in range(self.num_tasks):
            cur_data = data[i]
            cur_mixing = latent_mixing[i]
            cur_kl = self._compute_latent_kl_per_task(i, cur_data, cur_mixing)
            latents_kl.append(cur_kl)
        # average kl among tasks
        latents_kl = torch.stack(latents_kl).mean()
        
        # get classifiers kl for each task
        classifiers_kl = []
        for i in range(self.num_tasks):
            cur_mixing = classifier_mixing[i]
            cur_kl = self._compute_cls_kl_per_task(i, cur_mixing)
            classifiers_kl.append(cur_kl)
        # average kl among tasks
        classifiers_kl = torch.stack(classifiers_kl).mean()

        elbo = lh_val + latents_kl + classifiers_kl

        return {
            "elbo": elbo,
            "lh_loss": lh_val,
            "lat_kl": latents_kl,
            "cls_kl": classifiers_kl
        }

    def _compute_lh_per_task(
        self,
        task_num: int,
        latents: torch.Tensor,
        classifiers: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute -log prob for each latent and classifier particle for each batch,
        mean across classifiers and latents, sum across targets with batch size correction
        """
        task_cond_distr = self.task_distrs[task_num]
        target_shape = targets.shape[1:]
        batch_size = targets.shape[0]

        # log_prob shape=(batch_size, lat_num_part, classifier_num_part, target_shape)
        return -task_cond_distr(latents, classifiers).log_prob(
                targets[:, None, None, ...].expand(-1, self.latent_num_particles, self.classifier_num_particles, *target_shape)
            ).mean(dim=(1, 2)).sum(dim=0) * (self.task_num_samples[task_num] / batch_size)

    def _compute_latent_kl_per_task(
        self,
        task_num: int,
        inputs: torch.Tensor,
        latent_mixing: torch.Tensor
    ) -> torch.Tensor:
        batch_size = inputs.shape[0]
        cur_distr = self.latent_distr[task_num](inputs)

        return torch.stack(
            [self._compute_kl(cur_distr, lat_cond_distr(inputs)) for lat_cond_distr in self.latent_distr],
            dim=1
        ).matmul(latent_mixing).sum() * \
            (self.task_num_samples[task_num] / batch_size) # sum across batch with batch size correction

    def _compute_cls_kl_per_task(
        self,
        task_num: int,
        clas_mixing: torch.Tensor
    ) -> torch.Tensor:
        cur_distr = self.classifier_distr[task_num]

        return torch.stack(
            [self._compute_kl(cur_distr, cl_cond_distr) for cl_cond_distr in self.classifier_distr]
        ).dot(clas_mixing)

    def _compute_kl(self, distr_1: distr.Distribution, distr_2: distr.Distribution) -> torch.Tensor:
        """Computes KL analytically if possible else make a sample estimation
        """
        if distr_1 is distr_2:
            return torch.zeros(distr_1.batch_shape)

        try:
            return distr.kl_divergence(distr_1, distr_2)
        except NotImplementedError:
            return kl_sample_estimation(distr_1, distr_2, self.kl_estimator_num_samples)

    def _get_gumbelsm_mixing(self, mixings_params: torch.Tensor, temp: float) -> torch.Tensor:
        # mixing with self is prohibited, so we mask diagonal to get zeros after softmax
        mask = torch.diag(torch.full((self.num_tasks, ), -torch.inf))

        mixing = distr.Gumbel(0., 1.).sample((self.num_tasks, self.num_tasks))
        mixing += mixings_params.log()
        mixing = mixing / temp
        mixing += mask
        mixing = torch.softmax(mixing, dim=1)

        return mixing

    @property
    def classifier_mixings_params(self):
        """Accesses classifer mixing params
        """
        return self._classifier_mixings_params

    @property
    def latent_mixings_params(self):
        """Accesses latent mixing params
        """
        return self._latent_mixings_params
