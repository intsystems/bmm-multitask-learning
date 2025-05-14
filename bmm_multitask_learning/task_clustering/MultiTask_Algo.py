from MultiTask_Base_Class import MultiTaskNNBase
from MultiTask_Base_Class import SufficientStatistics
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from scipy.linalg import solve_triangular, cholesky
from tqdm import tqdm
from typing import List, Tuple, Optional

class MultiTaskNN(MultiTaskNNBase):
    """Basic multi-task neural network with shared hidden layer and task-specific output weights"""

    def __init__(self, n_input: int, n_hidden: int, n_tasks: int,
                 activation: str = 'tanh', regularization: float = 1e-6):
        super().__init__(n_input, n_hidden, n_tasks, activation, regularization)

        # Initialize hyperparameters
        self.m = np.random.randn(n_hidden + 1) * 0.5
        self.Sigma = np.eye(n_hidden + 1) * 0.5
        self.sigma = 1.0

    def log_likelihood(self, params: np.ndarray, all_stats: List[SufficientStatistics]) -> float:
        """Compute the log likelihood with numerical stability

        Args:
            params: Flattened array of all parameters
            all_stats: List of sufficient statistics for each task

        Returns:
            Total log likelihood across all tasks
        """
        try:
            # Unpack parameters
            param_idx = 0

            # W
            W_size = self.n_hidden * (self.n_input + 1)
            W = params[param_idx:param_idx + W_size].reshape(self.n_hidden, self.n_input + 1)
            param_idx += W_size

            # m
            m_size = self.n_hidden + 1
            m = params[param_idx:param_idx + m_size]
            param_idx += m_size

            # Sigma (Cholesky decomposition)
            L = np.zeros((self.n_hidden + 1, self.n_hidden + 1))
            tril_indices = np.tril_indices(self.n_hidden + 1)
            L[tril_indices] = params[param_idx:param_idx + len(tril_indices[0])]
            param_idx += len(tril_indices[0])

            # sigma (log scale)
            log_sigma = params[param_idx]
            sigma = np.exp(log_sigma)

            total_log_lik = 0.0
            self.A_map = []

            # Add regularization to Sigma
            Sigma = np.dot(L, L.T) + self.regularization * np.eye(self.n_hidden + 1)

            # Precompute Sigma inverse using Cholesky
            try:
                L_sigma = cholesky(Sigma, lower=True)
                Sigma_inv = solve_triangular(L_sigma, np.eye(self.n_hidden + 1), lower=True)
                Sigma_inv = np.dot(Sigma_inv.T, Sigma_inv)
            except np.linalg.LinAlgError:
                return -np.inf

            for stats in all_stats:
                # Compute Q_i with regularization
                Q_i = (1.0 / (sigma ** 2 + 1e-8)) * stats.sum_hhT + Sigma_inv

                try:
                    L_Q = cholesky(Q_i + self.regularization * np.eye(self.n_hidden + 1), lower=True)
                    Q_inv = solve_triangular(L_Q, np.eye(self.n_hidden + 1), lower=True)
                    Q_inv = np.dot(Q_inv.T, Q_inv)
                except np.linalg.LinAlgError:
                    return -np.inf

                R_i = (1.0 / (sigma ** 2 + 1e-8)) * stats.sum_hy + np.dot(Sigma_inv, m)

                # Compute MAP estimate with regularization
                A_i = np.linalg.solve(Q_i + self.regularization * np.eye(self.n_hidden + 1), R_i)
                self.A_map.append(A_i)

                # Compute log determinants
                logdet_Q_i = 2 * np.sum(np.log(np.diag(L_Q)))
                logdet_Sigma = 2 * np.sum(np.log(np.diag(L_sigma)))

                # Compute log likelihood terms
                term1 = -0.5 * (logdet_Sigma + stats.n_samples * 2 * log_sigma + logdet_Q_i)
                term2 = 0.5 * (
                        np.dot(R_i, np.dot(Q_inv, R_i)) - (1.0 / (sigma ** 2 + 1e-8)) * stats.sum_yy -
                        np.dot(m, np.dot(Sigma_inv, m))
                )

                if not np.isfinite(term1 + term2):
                    return -np.inf

                total_log_lik += term1 + term2

            return total_log_lik if np.isfinite(total_log_lik) else -np.inf

        except:
            return -np.inf

    def fit(self, X_list: List[np.ndarray], y_list: List[np.ndarray], max_iter: int = 100) -> Optional[object]:
        """Fit the model to data

        Args:
            X_list: List of input arrays for each task
            y_list: List of target arrays for each task
            max_iter: Maximum number of optimization iterations

        Returns:
            The optimization result or None if optimization failed
        """
        # Normalize data
        X_list, y_list = self._normalize_data(X_list, y_list)

        # Compute sufficient statistics
        all_stats = [self.compute_sufficient_statistics(X, y) for X, y in zip(X_list, y_list)]

        # Initial parameters with better scaling
        initial_params = []
        initial_params.extend(self.W.flatten())
        initial_params.extend(self.m)

        # Initialize Sigma with Cholesky decomposition
        L = np.linalg.cholesky(self.Sigma + self.regularization * np.eye(self.n_hidden + 1))
        tril_indices = np.tril_indices(self.n_hidden + 1)
        initial_params.extend(L[tril_indices])

        initial_params.append(np.log(self.sigma))

        # Optimize with bounds for stability
        bounds = []
        bounds.extend([(None, None)] * (self.n_hidden * (self.n_input + 1)))  # W
        bounds.extend([(None, None)] * (self.n_hidden + 1))  # m

        # L - diagonal elements must be positive
        for i in range(len(tril_indices[0])):
            if tril_indices[0][i] == tril_indices[1][i]:  # diagonal
                bounds.append((1e-8, None))
            else:
                bounds.append((None, None))

        bounds.append((np.log(1e-8), None))  # log_sigma

        # Optimization with error handling
        try:
            result = minimize(
                lambda p: -self.log_likelihood(p, all_stats),
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': max_iter,
                    'disp': True,
                    'maxfun': 15000,
                    'maxls': 50
                }
            )

            # Store optimized parameters
            self._unpack_parameters(result.x)

            # Recompute MAP estimates
            _ = self.log_likelihood(result.x, all_stats)

            return result

        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return None

    def _unpack_parameters(self, params: np.ndarray) -> None:
        """Helper to unpack optimized parameters

        Args:
            params: Flattened array of all parameters
        """
        param_idx = 0

        # W
        W_size = self.n_hidden * (self.n_input + 1)
        self.W = params[param_idx:param_idx + W_size].reshape(self.n_hidden, self.n_input + 1)
        param_idx += W_size

        # m
        m_size = self.n_hidden + 1
        self.m = params[param_idx:param_idx + m_size]
        param_idx += m_size

        # Sigma (Cholesky)
        L = np.zeros((self.n_hidden + 1, self.n_hidden + 1))
        tril_indices = np.tril_indices(self.n_hidden + 1)
        L[tril_indices] = params[param_idx:param_idx + len(tril_indices[0])]
        param_idx += len(tril_indices[0])
        self.Sigma = np.dot(L, L.T) + self.regularization * np.eye(self.n_hidden + 1)

        # sigma
        self.sigma = max(np.exp(params[param_idx]), 1e-8)


class MultiTaskNNDependentMean(MultiTaskNNBase):
    """Multi-task NN with task-dependent prior means"""

    def __init__(self, n_input: int, n_hidden: int, n_tasks: int, n_features: int,
                 activation: str = 'tanh', regularization: float = 1e-6):
        super().__init__(n_input, n_hidden, n_tasks, activation, regularization)
        self.n_features = n_features

        # Initialize hyperparameters with better scaling
        self.M = np.random.randn(n_hidden + 1, n_features) * 0.1
        self.Sigma = np.eye(n_hidden + 1) * 0.5
        self.sigma = 1.0

    def log_likelihood(self, params: np.ndarray, all_stats: List[SufficientStatistics],
                       all_task_features: List[np.ndarray]) -> float:
        """Compute the log likelihood with numerical stability improvements

        Args:
            params: Flattened array of all parameters
            all_stats: List of sufficient statistics for each task
            all_task_features: List of task feature vectors

        Returns:
            Total log likelihood across all tasks
        """
        # Unpack parameters
        param_idx = 0

        # W
        W_size = self.n_hidden * (self.n_input + 1)
        W = params[param_idx:param_idx + W_size].reshape(self.n_hidden, self.n_input + 1)
        param_idx += W_size

        # M: (n_hidden + 1 x n_features)
        M_size = (self.n_hidden + 1) * self.n_features
        M = params[param_idx:param_idx + M_size].reshape(self.n_hidden + 1, self.n_features)
        param_idx += M_size

        # Sigma (Cholesky decomposition)
        L = np.zeros((self.n_hidden + 1, self.n_hidden + 1))
        tril_indices = np.tril_indices(self.n_hidden + 1)
        L[tril_indices] = params[param_idx:param_idx + len(tril_indices[0])]
        param_idx += len(tril_indices[0])

        # sigma (log scale)
        log_sigma = params[param_idx]
        sigma = np.exp(log_sigma)

        total_log_lik = 0.0
        self.A_map = []

        # Precompute Sigma inverse using Cholesky
        try:
            Sigma = np.dot(L, L.T) + self.regularization * np.eye(self.n_hidden + 1)
            L_sigma = cholesky(Sigma, lower=True)
            Sigma_inv = solve_triangular(L_sigma, np.eye(self.n_hidden + 1), lower=True)
            Sigma_inv = np.dot(Sigma_inv.T, Sigma_inv)
        except np.linalg.LinAlgError:
            return -np.inf  # Invalid covariance matrix

        for stats, task_features in zip(all_stats, all_task_features):
            # Compute task-dependent prior mean
            m_i = np.dot(M, task_features)

            # Compute Q_i using Cholesky for stability
            Q_i = (1.0 / (sigma ** 2 + 1e-8)) * stats.sum_hhT + Sigma_inv

            try:
                L_Q = cholesky(Q_i + self.regularization * np.eye(self.n_hidden + 1), lower=True)
                Q_inv = solve_triangular(L_Q, np.eye(self.n_hidden + 1), lower=True)
                Q_inv = np.dot(Q_inv.T, Q_inv)
            except np.linalg.LinAlgError:
                return -np.inf

            R_i = (1.0 / (sigma ** 2 + 1e-8)) * stats.sum_hy + np.dot(Sigma_inv, m_i)

            # Compute MAP estimate
            A_i = np.dot(Q_inv, R_i)
            self.A_map.append(A_i)

            # Compute log determinants efficiently
            logdet_Q_i = 2 * np.sum(np.log(np.diag(L_Q)))
            logdet_Sigma = 2 * np.sum(np.log(np.diag(L_sigma)))

            # Compute log likelihood terms
            term1 = -0.5 * (logdet_Sigma + stats.n_samples * 2 * log_sigma + logdet_Q_i)
            term2 = 0.5 * (
                    np.dot(R_i, np.dot(Q_inv, R_i)) - (1.0 / (sigma ** 2 + 1e-8)) * stats.sum_yy -
                    np.dot(m_i, np.dot(Sigma_inv, m_i)))

            total_log_lik += term1 + term2

        return total_log_lik

    def fit(self, X_list: List[np.ndarray], y_list: List[np.ndarray],
            task_features_list: List[np.ndarray], max_iter: int = 100) -> object:
        """Fit the model with improved optimization

        Args:
            X_list: List of input arrays for each task
            y_list: List of target arrays for each task
            task_features_list: List of task feature vectors
            max_iter: Maximum number of optimization iterations

        Returns:
            The optimization result
        """
        # Normalize data
        X_list, y_list = self._normalize_data(X_list, y_list)

        # Compute sufficient statistics
        all_stats = [self.compute_sufficient_statistics(X, y) for X, y in zip(X_list, y_list)]

        # Initial parameters with better scaling
        initial_params = []
        initial_params.extend(self.W.flatten())
        initial_params.extend(self.M.flatten())

        L = np.linalg.cholesky(self.Sigma + self.regularization * np.eye(self.n_hidden + 1))
        tril_indices = np.tril_indices(self.n_hidden + 1)
        initial_params.extend(L[tril_indices])

        initial_params.append(np.log(self.sigma))

        # Optimize with bounds for stability
        bounds = []

        # W - no bounds
        bounds.extend([(None, None)] * (self.n_hidden * (self.n_input + 1)))

        # M - no bounds
        bounds.extend([(None, None)] * ((self.n_hidden + 1) * self.n_features))

        # L - diagonal elements must be positive
        for i in range(len(tril_indices[0])):
            if tril_indices[0][i] == tril_indices[1][i]:  # diagonal
                bounds.append((1e-8, None))
            else:
                bounds.append((None, None))

        # log_sigma must be > log(1e-8)
        bounds.append((np.log(1e-8), None))

        # Optimize
        result = minimize(
            lambda p: -self.log_likelihood(p, all_stats, task_features_list),
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': True}
        )

        # Store optimized parameters
        self._unpack_parameters(result.x)

        # Recompute MAP estimates
        _ = self.log_likelihood(result.x, all_stats, task_features_list)

        return result

    def _unpack_parameters(self, params: np.ndarray) -> None:
        """Helper to unpack optimized parameters

        Args:
            params: Flattened array of all parameters
        """
        param_idx = 0

        # W
        W_size = self.n_hidden * (self.n_input + 1)
        self.W = params[param_idx:param_idx + W_size].reshape(self.n_hidden, self.n_input + 1)
        param_idx += W_size

        # M: (n_hidden + 1 x n_features)
        M_size = (self.n_hidden + 1) * self.n_features
        self.M = params[param_idx:param_idx + M_size].reshape(self.n_hidden + 1, self.n_features)
        param_idx += M_size

        # Sigma (Cholesky)
        L = np.zeros((self.n_hidden + 1, self.n_hidden + 1))
        tril_indices = np.tril_indices(self.n_hidden + 1)
        L[tril_indices] = params[param_idx:param_idx + len(tril_indices[0])]
        param_idx += len(tril_indices[0])
        self.Sigma = np.dot(L, L.T) + self.regularization * np.eye(self.n_hidden + 1)

        # sigma
        self.sigma = np.exp(params[param_idx])


class MultiTaskNNClustering(MultiTaskNNBase):
    """Multi-task NN with task clustering"""

    def __init__(self, n_input: int, n_hidden: int, n_tasks: int, n_clusters: int,
                 activation: str = 'tanh', regularization: float = 1e-6):
        super().__init__(n_input, n_hidden, n_tasks, activation, regularization)
        self.n_clusters = n_clusters

        # Initialize with larger scale and better conditioning
        self.q = np.ones(n_clusters) / n_clusters
        self.m = np.random.randn(n_clusters, n_hidden + 1) * 0.5

        # Initialize Sigma with larger diagonal for numerical stability
        self.Sigma = np.array([np.eye(n_hidden + 1) * 0.5 for _ in range(n_clusters)])
        self.sigma = 1.0

        self.z = np.zeros((n_tasks, n_clusters))

    def _compute_task_log_likelihood(self, X_i: np.ndarray, y_i: np.ndarray,
                                     cluster_idx: int) -> float:
        """Compute log likelihood for a single task and cluster

        Args:
            X_i: Input data for the task
            y_i: Target values for the task
            cluster_idx: Index of cluster to compute for

        Returns:
            Log likelihood for the task under the specified cluster
        """
        n_i = len(y_i)
        h_i = self.compute_hidden_activations(X_i)

        # Add small constant to avoid division by zero
        sigma_sq = max(self.sigma ** 2, 1e-8)

        try:
            # Use Cholesky decomposition for numerical stability
            L = cholesky(self.Sigma[cluster_idx] + self.regularization * np.eye(self.n_hidden + 1), lower=True)
            Sigma_inv = solve_triangular(L, np.eye(self.n_hidden + 1), lower=True)
            Sigma_inv = np.dot(Sigma_inv.T, Sigma_inv)

            Q_i = (1 / sigma_sq) * np.dot(h_i.T, h_i) + Sigma_inv
            L_Q = cholesky(Q_i + self.regularization * np.eye(self.n_hidden + 1), lower=True)
            Q_inv = solve_triangular(L_Q, np.eye(self.n_hidden + 1), lower=True)
            Q_inv = np.dot(Q_inv.T, Q_inv)

            R_i = (1 / sigma_sq) * np.dot(h_i.T, y_i) + np.dot(Sigma_inv, self.m[cluster_idx])

            # Compute log determinants efficiently
            logdet_Sigma = 2 * np.sum(np.log(np.diag(L)))
            logdet_Q_i = 2 * np.sum(np.log(np.diag(L_Q)))

            term1 = -0.5 * (logdet_Sigma + n_i * np.log(sigma_sq) + logdet_Q_i)
            term2 = 0.5 * (
                    np.dot(R_i.T, np.dot(Q_inv, R_i)) -
                    (1 / (2 * sigma_sq)) * np.sum(y_i ** 2) -
                    np.dot(self.m[cluster_idx].T, np.dot(Sigma_inv, self.m[cluster_idx]))
            )

            return term1 + term2

        except np.linalg.LinAlgError:
            # Return -inf if matrix is not positive definite
            return -np.inf

    def e_step(self, data: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Expectation step - compute cluster responsibilities

        Args:
            data: List of (X, y) tuples for each task
        """
        log_responsibilities = np.zeros((self.n_tasks, self.n_clusters))

        for i, (X_i, y_i) in enumerate(data):
            for alpha in range(self.n_clusters):
                log_lik = self._compute_task_log_likelihood(X_i, y_i, alpha)
                log_responsibilities[i, alpha] = np.log(self.q[alpha] + 1e-8) + log_lik

            # Normalize using logsumexp for numerical stability
            log_responsibilities[i] -= logsumexp(log_responsibilities[i])

        self.z = np.exp(log_responsibilities)

    def m_step(self, data: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Maximization step - update model parameters

        Args:
            data: List of (X, y) tuples for each task
        """

        def objective(params: np.ndarray) -> float:
            """Objective function for optimizing W and sigma"""
            W = params[:self.n_hidden * (self.n_input + 1)].reshape(self.n_hidden, self.n_input + 1)
            log_sigma = params[-1]
            sigma = np.exp(log_sigma)

            self.W = W
            self.sigma = max(sigma, 1e-8)  # Prevent sigma from becoming too small

            total_log_lik = 0.0
            for i, (X_i, y_i) in enumerate(data):
                for alpha in range(self.n_clusters):
                    log_lik = self._compute_task_log_likelihood(X_i, y_i, alpha)
                    total_log_lik += self.z[i, alpha] * log_lik

            return -total_log_lik if np.isfinite(total_log_lik) else np.inf

        # Initial parameters with bounds
        initial_params = np.concatenate([
            self.W.flatten(),
            [np.log(self.sigma)]
        ])

        # Add bounds for sigma (log_sigma > log(1e-8))
        bounds = [(None, None)] * len(initial_params)
        bounds[-1] = (np.log(1e-8), None)

        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'disp': True}
        )

        opt_params = result.x
        W_size = self.n_hidden * (self.n_input + 1)
        self.W = opt_params[:W_size].reshape(self.n_hidden, self.n_input + 1)
        self.sigma = max(np.exp(opt_params[-1]), 1e-8)

        # Update cluster parameters with regularization
        for alpha in range(self.n_clusters):
            # Update cluster probabilities
            self.q[alpha] = max(np.sum(self.z[:, alpha]) / self.n_tasks, 1e-8)

            sum_z = np.sum(self.z[:, alpha])
            if sum_z > 1e-8:
                weighted_R = np.zeros(self.n_hidden + 1)
                weighted_Q = np.zeros((self.n_hidden + 1, self.n_hidden + 1))

                for i, (X_i, y_i) in enumerate(data):
                    h_i = self.compute_hidden_activations(X_i)
                    L = cholesky(self.Sigma[alpha] + self.regularization * np.eye(self.n_hidden + 1), lower=True)
                    Sigma_inv = solve_triangular(L, np.eye(self.n_hidden + 1), lower=True)
                    Sigma_inv = np.dot(Sigma_inv.T, Sigma_inv)

                    Q_i = (1 / max(self.sigma ** 2, 1e-8)) * np.dot(h_i.T, h_i) + Sigma_inv
                    R_i = (1 / max(self.sigma ** 2, 1e-8)) * np.dot(h_i.T, y_i) + np.dot(Sigma_inv, self.m[alpha])

                    weighted_R += self.z[i, alpha] * R_i
                    weighted_Q += self.z[i, alpha] * Q_i

                try:
                    # Update cluster mean
                    self.m[alpha] = np.linalg.solve(
                        weighted_Q + self.regularization * np.eye(self.n_hidden + 1),
                        weighted_R
                    )
                except np.linalg.LinAlgError:
                    continue

                # Update cluster covariance
                weighted_cov = np.zeros((self.n_hidden + 1, self.n_hidden + 1))
                for i, (X_i, y_i) in enumerate(data):
                    h_i = self.compute_hidden_activations(X_i)
                    A_i = self._compute_map_estimate(X_i, y_i, alpha)
                    diff = A_i - self.m[alpha]
                    weighted_cov += self.z[i, alpha] * np.outer(diff, diff)

                self.Sigma[alpha] = weighted_cov / sum_z + self.regularization * np.eye(self.n_hidden + 1)

    def _compute_map_estimate(self, X_i: np.ndarray, y_i: np.ndarray,
                              cluster_idx: int) -> np.ndarray:
        """Compute MAP estimate for a task under a specific cluster

        Args:
            X_i: Input data for the task
            y_i: Target values for the task
            cluster_idx: Index of cluster to compute for

        Returns:
            MAP estimate of output weights
        """
        h_i = self.compute_hidden_activations(X_i)
        sigma_sq = max(self.sigma ** 2, 1e-8)

        L = cholesky(self.Sigma[cluster_idx] + self.regularization * np.eye(self.n_hidden + 1), lower=True)
        Sigma_inv = solve_triangular(L, np.eye(self.n_hidden + 1), lower=True)
        Sigma_inv = np.dot(Sigma_inv.T, Sigma_inv)

        Q_i = (1 / sigma_sq) * np.dot(h_i.T, h_i) + Sigma_inv
        R_i = (1 / sigma_sq) * np.dot(h_i.T, y_i) + np.dot(Sigma_inv, self.m[cluster_idx])

        return np.linalg.solve(Q_i + self.regularization * np.eye(self.n_hidden + 1), R_i)

    def fit(self, data: List[Tuple[np.ndarray, np.ndarray]],
            max_iter: int = 100, tol: float = 1e-4) -> None:
        """Fit the model using EM algorithm

        Args:
            data: List of (X, y) tuples for each task
            max_iter: Maximum number of EM iterations
            tol: Convergence tolerance for log likelihood
        """
        prev_log_lik = -np.inf

        for iteration in tqdm(range(max_iter)):
            self.e_step(data)
            self.m_step(data)

            # Compute current log likelihood
            current_log_lik = 0.0
            for i, (X_i, y_i) in enumerate(data):
                cluster_log_liks = []
                for alpha in range(self.n_clusters):
                    log_lik = self._compute_task_log_likelihood(X_i, y_i, alpha)
                    cluster_log_liks.append(np.log(self.q[alpha] + 1e-8) + log_lik)
                current_log_lik += logsumexp(cluster_log_liks)

            if np.isnan(current_log_lik):
                print("Warning: log likelihood is nan, stopping early")
                break

            if iteration > 0 and np.abs(current_log_lik - prev_log_lik) < tol:
                print(f"Converged at iteration {iteration}")
                break

            prev_log_lik = current_log_lik

        self._compute_final_weights(data)

    def _compute_final_weights(self, data: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Compute final output weights for each task

        Args:
            data: List of (X, y) tuples for each task
        """
        for i, (X_i, y_i) in enumerate(data):
            most_likely_cluster = np.argmax(self.z[i])
            self.A_map[i] = self._compute_map_estimate(X_i, y_i, most_likely_cluster)

    def get_cluster_assignments(self) -> np.ndarray:
        """Get cluster assignments for each task

        Returns:
            Array of cluster indices for each task
        """
        return np.argmax(self.z, axis=1)

    def get_task_similarity(self) -> np.ndarray:
        """Get task similarity matrix based on cluster assignments

        Returns:
            Similarity matrix where 1 indicates same cluster, 0 different
        """
        assignments = self.get_cluster_assignments()
        return np.array([[1.0 if a == b else 0.0 for b in assignments] for a in assignments])


class MultiTaskNNGating(MultiTaskNNBase):
    """Multi-task NN with gating network for task clustering"""

    def __init__(self, n_input: int, n_hidden: int, n_tasks: int, n_clusters: int,
                 n_features: int, activation: str = 'tanh', regularization: float = 1e-6):
        super().__init__(n_input, n_hidden, n_tasks, activation, regularization)
        self.n_clusters = n_clusters
        self.n_features = n_features

        # Initialize with larger scale for better convergence
        self.U = np.random.randn(n_clusters, n_features) * 0.5
        self.m = np.random.randn(n_clusters, n_hidden + 1) * 0.5

        # Initialize covariance matrices with larger diagonal
        self.Sigma = np.array([np.eye(n_hidden + 1) * 0.5 for _ in range(n_clusters)])
        self.sigma = 1.0

        self.z = np.zeros((n_tasks, n_clusters))

        # Task feature normalization parameters
        self.task_feature_mean = None
        self.task_feature_std = None

    def compute_gating_probabilities(self, F: np.ndarray) -> np.ndarray:
        """Compute task-cluster assignment probabilities with numerical stability

        Args:
            F: Task features (n_tasks x n_features)

        Returns:
            Probability matrix (n_tasks x n_clusters)
        """
        # Ensure F is 2D array
        F = np.atleast_2d(F)
        if F.shape[0] == 1 and self.n_tasks > 1:
            F = np.repeat(F, self.n_tasks, axis=0)

        # Normalize if normalization parameters are available
        if self.task_feature_mean is not None and self.task_feature_std is not None:
            F = (F - self.task_feature_mean) / (self.task_feature_std + 1e-8)

        logits = np.dot(F, self.U.T)
        return softmax(logits, axis=1)

    def _compute_task_log_likelihood(self, X_i: np.ndarray, y_i: np.ndarray,
                                     cluster_idx: int) -> float:
        """Compute log likelihood for a single task and cluster

        Args:
            X_i: Input data for the task
            y_i: Target values for the task
            cluster_idx: Index of cluster to compute for

        Returns:
            Log likelihood for the task under the specified cluster
        """
        n_i = len(y_i)
        h_i = self.compute_hidden_activations(X_i)

        # Add small constant to avoid division by zero
        sigma_sq = max(self.sigma ** 2, 1e-8)

        try:
            # Use Cholesky decomposition for numerical stability
            L = cholesky(self.Sigma[cluster_idx] + self.regularization * np.eye(self.n_hidden + 1), lower=True)
            Sigma_inv = solve_triangular(L, np.eye(self.n_hidden + 1), lower=True)
            Sigma_inv = np.dot(Sigma_inv.T, Sigma_inv)

            Q_i = (1 / sigma_sq) * np.dot(h_i.T, h_i) + Sigma_inv
            L_Q = cholesky(Q_i + self.regularization * np.eye(self.n_hidden + 1), lower=True)
            Q_inv = solve_triangular(L_Q, np.eye(self.n_hidden + 1), lower=True)
            Q_inv = np.dot(Q_inv.T, Q_inv)

            R_i = (1 / sigma_sq) * np.dot(h_i.T, y_i) + np.dot(Sigma_inv, self.m[cluster_idx])

            # Compute log determinants
            logdet_Sigma = 2 * np.sum(np.log(np.diag(L)))
            logdet_Q_i = 2 * np.sum(np.log(np.diag(L_Q)))

            term1 = -0.5 * (logdet_Sigma + n_i * np.log(sigma_sq) + logdet_Q_i)
            term2 = 0.5 * (
                    np.dot(R_i.T, np.dot(Q_inv, R_i)) -
                    (1 / (2 * sigma_sq)) * np.sum(y_i ** 2) -
                    np.dot(self.m[cluster_idx].T, np.dot(Sigma_inv, self.m[cluster_idx]))
            )

            return term1 + term2

        except np.linalg.LinAlgError:
            return -np.inf

    def e_step(self, data: List[Tuple[np.ndarray, np.ndarray]],
               task_features: np.ndarray) -> None:
        """Expectation step with improved numerical stability

        Args:
            data: List of (X, y) tuples for each task
            task_features: Task features (n_tasks x n_features)
        """
        # Ensure task_features is 2D array
        task_features = np.atleast_2d(task_features)
        if task_features.shape[0] == 1 and self.n_tasks > 1:
            task_features = np.repeat(task_features, self.n_tasks, axis=0)

        q = self.compute_gating_probabilities(task_features)
        log_responsibilities = np.zeros((self.n_tasks, self.n_clusters))

        for i, (X_i, y_i) in enumerate(data):
            for alpha in range(self.n_clusters):
                log_lik = self._compute_task_log_likelihood(X_i, y_i, alpha)
                log_responsibilities[i, alpha] = np.log(q[i, alpha] + 1e-8) + log_lik

            # Normalize using logsumexp
            log_responsibilities[i] -= logsumexp(log_responsibilities[i])

        self.z = np.exp(log_responsibilities)

    def m_step(self, data: List[Tuple[np.ndarray, np.ndarray]],
               task_features: np.ndarray) -> None:
        """Maximization step with regularization

        Args:
            data: List of (X, y) tuples for each task
            task_features: Task features (n_tasks x n_features)
        """

        # Optimize W and sigma
        def objective(params: np.ndarray) -> float:
            """Objective function for optimizing W and sigma"""
            W = params[:self.n_hidden * (self.n_input + 1)].reshape(self.n_hidden, self.n_input + 1)
            log_sigma = params[-1]
            sigma = np.exp(log_sigma)

            self.W = W
            self.sigma = max(sigma, 1e-8)

            total_log_lik = 0.0
            for i, (X_i, y_i) in enumerate(data):
                for alpha in range(self.n_clusters):
                    log_lik = self._compute_task_log_likelihood(X_i, y_i, alpha)
                    total_log_lik += self.z[i, alpha] * log_lik

            return -total_log_lik if np.isfinite(total_log_lik) else np.inf

        # Initial parameters with bounds
        initial_params = np.concatenate([
            self.W.flatten(),
            [np.log(self.sigma)]
        ])

        bounds = [(None, None)] * len(initial_params)
        bounds[-1] = (np.log(1e-8), None)  # sigma > 1e-8

        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'disp': True}
        )

        # Update parameters
        opt_params = result.x
        W_size = self.n_hidden * (self.n_input + 1)
        self.W = opt_params[:W_size].reshape(self.n_hidden, self.n_input + 1)
        self.sigma = max(np.exp(opt_params[-1]), 1e-8)

        # Update cluster parameters with regularization
        for alpha in range(self.n_clusters):
            sum_z = np.sum(self.z[:, alpha])
            if sum_z > 1e-8:
                # Update m_α
                weighted_R = np.zeros(self.n_hidden + 1)
                weighted_Q = np.zeros((self.n_hidden + 1, self.n_hidden + 1))

                for i, (X_i, y_i) in enumerate(data):
                    h_i = self.compute_hidden_activations(X_i)
                    L = cholesky(self.Sigma[alpha] + self.regularization * np.eye(self.n_hidden + 1), lower=True)
                    Sigma_inv = solve_triangular(L, np.eye(self.n_hidden + 1), lower=True)
                    Sigma_inv = np.dot(Sigma_inv.T, Sigma_inv)

                    Q_i = (1 / max(self.sigma ** 2, 1e-8)) * np.dot(h_i.T, h_i) + Sigma_inv
                    R_i = (1 / max(self.sigma ** 2, 1e-8)) * np.dot(h_i.T, y_i) + np.dot(Sigma_inv, self.m[alpha])

                    weighted_R += self.z[i, alpha] * R_i
                    weighted_Q += self.z[i, alpha] * Q_i

                try:
                    self.m[alpha] = np.linalg.solve(
                        weighted_Q + self.regularization * np.eye(self.n_hidden + 1),
                        weighted_R
                    )
                except np.linalg.LinAlgError:
                    pass

                # Update Σ_α with regularization
                weighted_cov = np.zeros((self.n_hidden + 1, self.n_hidden + 1))
                for i, (X_i, y_i) in enumerate(data):
                    h_i = self.compute_hidden_activations(X_i)
                    A_i = self._compute_map_estimate(X_i, y_i, alpha)
                    diff = A_i - self.m[alpha]
                    weighted_cov += self.z[i, alpha] * np.outer(diff, diff)

                self.Sigma[alpha] = weighted_cov / sum_z + self.regularization * np.eye(self.n_hidden + 1)

        # Update gating parameters U
        if self.n_clusters > 1:
            task_features = np.atleast_2d(task_features)
            if task_features.shape[0] == 1 and self.n_tasks > 1:
                task_features = np.repeat(task_features, self.n_tasks, axis=0)

            # Normalize task features if not already done
            if self.task_feature_mean is None or self.task_feature_std is None:
                self.task_feature_mean = np.mean(task_features, axis=0)
                self.task_feature_std = np.std(task_features, axis=0) + 1e-8
                task_features = (task_features - self.task_feature_mean) / self.task_feature_std

            lr = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                fit_intercept=False,
                max_iter=100,
                penalty='l2',
                C=1.0
            )
            try:
                lr.fit(task_features, self.get_cluster_assignments(), sample_weight=np.max(self.z, axis=1))
                self.U = lr.coef_
            except Exception as e:
                print(f"Warning: Failed to update gating parameters: {str(e)}")

    def _compute_map_estimate(self, X_i: np.ndarray, y_i: np.ndarray,
                              cluster_idx: int) -> np.ndarray:
        """Compute MAP estimate for a task under a specific cluster

        Args:
            X_i: Input data for the task
            y_i: Target values for the task
            cluster_idx: Index of cluster to compute for

        Returns:
            MAP estimate of output weights
        """
        h_i = self.compute_hidden_activations(X_i)
        sigma_sq = max(self.sigma ** 2, 1e-8)

        L = cholesky(self.Sigma[cluster_idx] + self.regularization * np.eye(self.n_hidden + 1), lower=True)
        Sigma_inv = solve_triangular(L, np.eye(self.n_hidden + 1), lower=True)
        Sigma_inv = np.dot(Sigma_inv.T, Sigma_inv)

        Q_i = (1 / sigma_sq) * np.dot(h_i.T, h_i) + Sigma_inv
        R_i = (1 / sigma_sq) * np.dot(h_i.T, y_i) + np.dot(Sigma_inv, self.m[cluster_idx])

        return np.linalg.solve(Q_i + self.regularization * np.eye(self.n_hidden + 1), R_i)

    def fit(self, data: List[Tuple[np.ndarray, np.ndarray]],
            task_features: np.ndarray, max_iter: int = 100, tol: float = 1e-4) -> 'MultiTaskNNGating':
        """Fit the model using EM algorithm

        Args:
            data: List of (X, y) tuples for each task
            task_features: Task features (n_tasks x n_features)
            max_iter: Maximum number of EM iterations
            tol: Convergence tolerance for log likelihood

        Returns:
            self for method chaining
        """
        prev_log_lik = -np.inf

        # Normalize task features
        task_features = np.atleast_2d(task_features)
        if task_features.shape[0] == 1 and self.n_tasks > 1:
            task_features = np.repeat(task_features, self.n_tasks, axis=0)

        self.task_feature_mean = np.mean(task_features, axis=0)
        self.task_feature_std = np.std(task_features, axis=0) + 1e-8
        task_features = (task_features - self.task_feature_mean) / self.task_feature_std

        for iteration in range(max_iter):
            try:
                self.e_step(data, task_features)
                self.m_step(data, task_features)

                # Compute current log likelihood
                current_log_lik = 0.0
                q = self.compute_gating_probabilities(task_features)

                for i, (X_i, y_i) in enumerate(data):
                    cluster_log_liks = []
                    for alpha in range(self.n_clusters):
                        log_lik = self._compute_task_log_likelihood(X_i, y_i, alpha)
                        cluster_log_liks.append(np.log(q[i, alpha] + 1e-8) + log_lik)
                    current_log_lik += logsumexp(cluster_log_liks)

                if np.isnan(current_log_lik):
                    print("Warning: log likelihood is nan, stopping early")
                    break

                if iteration > 0 and abs(current_log_lik - prev_log_lik) < tol:
                    print(f"Converged at iteration {iteration}")
                    break

                prev_log_lik = current_log_lik
                print(f"Iteration {iteration}, log likelihood: {current_log_lik}")

            except Exception as e:
                print(f"Error at iteration {iteration}: {str(e)}")
                break

        self._compute_final_weights(data)
        return self

    def _compute_final_weights(self, data: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Compute final output weights for each task

        Args:
            data: List of (X, y) tuples for each task
        """
        for i, (X_i, y_i) in enumerate(data):
            most_likely_cluster = np.argmax(self.z[i])
            self.A_map[i] = self._compute_map_estimate(X_i, y_i, most_likely_cluster)

    def get_cluster_assignments(self) -> np.ndarray:
        """Get cluster assignments for each task

        Returns:
            Array of cluster indices for each task
        """
        return np.argmax(self.z, axis=1)

    def get_task_similarity(self) -> np.ndarray:
        """Get task similarity matrix based on cluster assignments

        Returns:
            Similarity matrix where 1 indicates same cluster, 0 different
        """
        assignments = self.get_cluster_assignments()
        return np.array([[1.0 if a == b else 0.0 for b in assignments] for a in assignments])