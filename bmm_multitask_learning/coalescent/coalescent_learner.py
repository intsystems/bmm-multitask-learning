import numpy as np
from bmm_multitask_learning.coalescent.coalescent_inference import Item, CoalescentTree
from bmm_multitask_learning.coalescent.inverse_wishart import InverseWishart
from bmm_multitask_learning.coalescent.parameters_cov import optimal_R, optimal_cov
from bmm_multitask_learning.coalescent.task_classes import TaskData
from bmm_multitask_learning.coalescent.utils import grad_optimizer


class S_handler:
    """
    class to incapsulate the methods for updating covariance scalers S
    """
    def __init__(self, s_list):
        self.s_list = s_list
        self.K = len(self.s_list)

    def log_prob_S(self, s, R_inv, L_inv, P, W):
        """
        returns the log probability of proposed parameters for S
        """
        s_neg = -s
        s_max = np.max(s_neg)
        s_neg = s_neg - s_max
        s_exp = np.exp(s_neg) * np.exp(s_max)
        if len(s_exp.shape) > 1:
            s_exp = s_exp.squeeze()
        S_exp = np.diag(s_exp)
        
        S = np.diag(s)

        cov_m = S_exp @ R_inv @ S_exp
        return - np.trace(S) - 1/2 * np.trace((S - P) @ L_inv @ (S - P)) - 1/2 * np.trace(W @ cov_m @ W)

    def _optimize_s(self, R, L, w, p, s_0, verbose=False):
        """
        This method is to optimize for given parameters
        """
        L_inv = np.linalg.inv(L)
        R_inv = np.linalg.inv(R)
        W = np.diag(w)
        P = np.diag(p)
        d = R.shape[0]
        
        def grad_S(s):
            s_neg = -s
            s_max = np.max(s_neg)
            s_neg = s_neg - s_max
            s_exp = np.exp(s_neg) * np.exp(s_max)
            if len(s_exp.shape) > 1:
                s_exp = s_exp.squeeze()
            S_exp = np.diag(s_exp)

            S = np.diag(s)

            cov_m = S_exp @ R_inv @ S_exp
            return (- np.eye(d) - (S - P) @ L_inv + W @ cov_m @ W).diagonal()

        if verbose:
            print(f"[INFO] log prob start: {self.log_prob_S(s_0, R_inv, L_inv, P, W)}")

        optimizer = grad_optimizer(100, 0.001, grad_S)
        s_opt = optimizer.run(s_0)
        
        if verbose:
            print(f"[INFO] log prob trained: {self.log_prob_S(s_opt)}")

        return s_opt

    def update_param(self, R, L, weights,
                    coalescent_tree: CoalescentTree,
                    verbose=False):
        for i in range(self.K):
            parent_s = coalescent_tree.leaves[i].parent.mean
            w = weights[i]
            s_0 = self.s_list[i]
            self.s_list[i] = self._optimize_s(R, L, w, parent_s, s_0, verbose)


class MultitaskProblem:
    """
    bayessian optimizer for Multitask Learning based on Coalescent [1]

    To use initialize with list of TaskData and call run. Then trained weights 
    will be available by method get_weights()




    [1] @article{daume2009bayesian,
            title={Bayesian multitask learning with latent hierarchies},
            author={Daum{\'e} III, Hal},
            journal={arXiv preprint arXiv:0907.0783},
            year={2009}
        }

    """
    def __init__(self,
            tasks: list[TaskData],
            dim,
            rho=0.05,
            cov_sigma=0.1,
            s_init=None
        ):
        """
        
        :param tasks: list of TaskData, which will be learned
        :param dim: dimention of problems
        :param rho: parameter that scales noise level in labels
        :param cov_sigma: covariance matrix scaler for Coalescent evolution variation
        :param s_init: initial values for S_i: variance scales of data
        """

        self.tasks = tasks
        self.K = len(tasks)
        self.dim = dim
        self.rho = 0.05
        self.R_distr = InverseWishart(dim, dim+1, corr_mat=True)

        self.cov_sigma = cov_sigma
        self.L_distr = InverseWishart(dim, dim+1,  cov_sigma * np.eye(dim), corr_mat=False)

        if s_init is None:
            s_init = np.zeros((self.K, dim), dtype=float)
            for i in range(self.K):
                S = np.random.randn(dim)/5
                s_init[i] = S
        else:
            assert s_init.shape == (self.K, dim), f"provided s_init have\
                incorrect shape: {s_init.shape=} instead of {(self.K, dim)}"

        self.S_leaves: S_handler = S_handler(s_init)
        self.weights = np.zeros((self.K, dim), dtype=float)

        self._trained = False

    def get_weights(self):
        if not self._trained:
            raise Warning("first call fit() method of trainer.")
        return self.weights
    
    def fit(self, n_steps=100):
        assert n_steps > 0 and isinstance(n_steps, int), "number of steps should be positive integer"

        # first setup the weights as in simple linear regression
        self._update_weights(
            tasks=self.tasks,
            weights_mp=self.weights,
            S_leaves=None,
            R=None,
            K=self.K,
            rho=self.rho,
            cov=np.eye(self.dim)
        )

        # method iteration
        for _ in range(n_steps):
            # get the most probable parameters
            R = self.R_distr.get_most_prob()
            L = self.L_distr.get_most_prob()


        # inference on coalescent tree.
        # Integrate out the evoulution of covariance
            leaves = [Item(elem, cov=0) for elem in self.S_leaves.s_list]
            coalescent_tree = CoalescentTree(leaves, L, self.dim)

        # gradient optimization of S_leaves
        # based on generated tree and parameters
            self.S_leaves.update_param(R, L, self.weights, coalescent_tree)

        # update the posteriors based on new tree and weights
            # update covariance matrix posterior
            L_samples = optimal_cov(coalescent_tree, self.dim)
            self.L_distr.update_posterior(L_samples)


            # update correlation matrix_posterior
            R_samples = optimal_R(self.S_leaves.s_list, self.weights,)
            self.R_distr.update_posterior(R_samples)

            # update weights based on new posterior
            self._update_weights(
                tasks=self.tasks,
                weights_mp=self.weights,
                S_leaves=self.S_leaves.s_list,
                R=R,
                K=self.K,
                rho=self.rho,
                cov=None
            )
 
        self._trained = True

    def _update_weights(self, tasks, weights_mp, S_leaves, R, K, rho, cov=None):
        for i in range(K):
            if cov is None:
                S = np.diag(S_leaves[i])
                cov_i = np.exp(S) @ R @ np.exp(S)
            else:
                cov_i = cov

            w = tasks[i].most_prob_w(cov_i, rho)
            weights_mp[i] = w
