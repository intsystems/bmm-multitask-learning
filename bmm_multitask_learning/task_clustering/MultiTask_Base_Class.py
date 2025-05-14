import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SufficientStatistics:
    """Container for sufficient statistics of a single task"""
    sum_hhT: np.ndarray  # Sum of H.T @ H
    sum_hy: np.ndarray  # Sum of H.T @ y
    sum_yy: float  # Sum of y.T @ y
    n_samples: int  # Number of samples


class MultiTaskNNBase:
    """Base class for all multi-task neural network variants

    Attributes:
        n_input: Number of input features
        n_hidden: Number of hidden units
        n_tasks: Number of tasks
        activation: Activation function ('tanh' or 'linear')
        W: Shared hidden layer weights
        A_map: Task-specific output weights
        regularization: Regularization strength for covariance matrices
    """

    def __init__(self, n_input: int, n_hidden: int, n_tasks: int,
                 activation: str = 'tanh', regularization: float = 1e-6):
        """
        Initialize base multi-task neural network

        Args:
            n_input: Number of input features
            n_hidden: Number of hidden units
            n_tasks: Number of tasks
            activation: Activation function ('tanh' or 'linear')
            regularization: Regularization strength for covariance matrices
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_tasks = n_tasks
        self.activation = activation
        self.regularization = regularization

        # Initialize weights and normalization parameters
        self._initialize_weights()
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def _initialize_weights(self, scale: float = 0.5) -> None:
        """Initialize network weights with given scale"""
        self.W = np.random.randn(self.n_hidden, self.n_input + 1) * scale
        self.A_map = [np.zeros(self.n_hidden + 1) for _ in range(self.n_tasks)]

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function to hidden units

        Args:
            x: Input array

        Returns:
            Activated output array
        """
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'linear':
            return x
        else:
            raise ValueError("Activation must be 'tanh' or 'linear'")

    def compute_hidden_activations(self, X: np.ndarray) -> np.ndarray:
        """Compute hidden unit activations with bias

        Args:
            X: Input data (n_samples x n_features)

        Returns:
            Hidden activations with bias term (n_samples x n_hidden+1)
        """
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        H = self._activate(np.dot(X_bias, self.W.T))
        return np.hstack([H, np.ones((H.shape[0], 1))])

    def predict(self, X: np.ndarray, task_idx: int) -> np.ndarray:
        """Make predictions for a specific task

        Args:
            X: Input data (n_samples x n_features)
            task_idx: Index of task to predict

        Returns:
            Predictions for the specified task
        """
        # Normalize input if normalization parameters are available
        if self.X_mean is not None and self.X_std is not None:
            X = (X - self.X_mean) / (self.X_std + 1e-8)

        H_bias = self.compute_hidden_activations(X)
        pred = np.dot(H_bias, self.A_map[task_idx])

        # Unnormalize output if normalization parameters are available
        if self.y_mean is not None and self.y_std is not None:
            pred = pred * (self.y_std[task_idx] + 1e-8) + self.y_mean[task_idx]

        return pred

    def compute_sufficient_statistics(self, X: np.ndarray, y: np.ndarray) -> SufficientStatistics:
        """Compute sufficient statistics for a single task

        Args:
            X: Input data (n_samples x n_features)
            y: Target values (n_samples)

        Returns:
            SufficientStatistics object containing computed statistics
        """
        H_bias = self.compute_hidden_activations(X)
        return SufficientStatistics(
            sum_hhT=np.dot(H_bias.T, H_bias),
            sum_hy=np.dot(H_bias.T, y),
            sum_yy=np.dot(y, y),
            n_samples=X.shape[0]
        )

    def _normalize_data(self, X_list: List[np.ndarray], y_list: List[np.ndarray]) -> Tuple[
        List[np.ndarray], List[np.ndarray]]:
        """Normalize input data and store normalization parameters

        Args:
            X_list: List of input arrays for each task
            y_list: List of target arrays for each task

        Returns:
            Tuple of (normalized_X_list, normalized_y_list)
        """
        # Compute and store normalization parameters for X
        all_X = np.vstack(X_list)
        self.X_mean = np.mean(all_X, axis=0)
        self.X_std = np.std(all_X, axis=0) + 1e-8
        X_norm = [(X - self.X_mean) / self.X_std for X in X_list]

        # Compute and store normalization parameters for y
        self.y_mean = [np.mean(y) for y in y_list]
        self.y_std = [np.std(y) + 1e-8 for y in y_list]
        y_norm = [(y - self.y_mean[i]) / self.y_std[i] for i, y in enumerate(y_list)]

        return X_norm, y_norm