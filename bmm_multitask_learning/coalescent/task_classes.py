from dataclasses import dataclass

import numpy as np


@dataclass
class TaskData:
    X: np.ndarray  # (n_task, d)
    y: np.ndarray  # (n_task, 1)

    def most_prob_w(self, sigma, rho):
        """
        :param sigma: cov matrix of prior of w
        :param rho: scaling coefficient for data
        """
        coeff = (1/rho**2)
        mean_est_uncent = coeff * (self.X * self.y).sum(0)
        cov_fixed = np.linalg.inv(np.linalg.inv(sigma) +
                            coeff * (self.X.T @ self.X))

        mean_est = cov_fixed @ (mean_est_uncent)
        return mean_est
