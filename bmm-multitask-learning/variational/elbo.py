from typing import Literal, Callable

import torch
from torch import nn
from torch import distributions as distr

from distr import kl_sample_estimation

class MultiTaskElbo(nn.Module):
    """General ELBO computer for variational multitask problem. 
    """
    def __init__(
        self,
        task_distrs: list[distr.Distribution],
        task_num_samples: list[int],
        classifier_distr: distr.Distribution,
        latent_distr: distr.Distribution,
        classifier_num_particles: int = 1,
        latent_num_particles: int = 1,
        temp_scheduler: Callable | Literal["const"] = Literal["const"]
    ):
        """
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

        self.temp_scheduler = temp_scheduler if temp_scheduler is not "const" else lambda t: 1.

        # define gumbel-softmax parameters for classifier and latent
        # initialize uniform
        self._classifier_mixings, self._latent_mixings = [
            nn.Parameter(
                torch.full((self.num_tasks, self.num_tasks - 1), 1 / (self.num_tasks - 1))
            )
        ] * 2

    def forward(self, targets: list[torch.Tensor], data: list[torch.Tensor]) -> torch.Tensor:
        """Computes ELBO estimation for variational multitask problem.

        Args:
            targets (list[torch.Tensor]): batched targets (y) for each task 
            data (list[torch.Tensor]): batched data (X) for each task 

        Returns:
            torch.Tensor: ELBO estimation
        """
        ...        

    @property
    def classifier_mixings(self):
        return self._classifier_mixings
    
    @property
    def latent_mixings(self):
        return self.latent_mixings