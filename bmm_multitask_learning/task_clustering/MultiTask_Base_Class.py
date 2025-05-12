import numpy as np
class MultiTaskNNBase:
    """Base class for all multi-task neural network variants"""

    def __init__(self, n_input, n_hidden, n_tasks, activation='tanh'):
        """
        Initialize base multi-task neural network

        Args:
            n_input: Number of input features
            n_hidden: Number of hidden units
            n_tasks: Number of tasks
            activation: Activation function ('tanh' or 'linear')
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_tasks = n_tasks
        self.activation = activation

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self, scale=0.5):
        """Initialize network weights with given scale"""
        self.W = np.random.randn(self.n_hidden, self.n_input + 1) * scale
        self.A_map = [np.zeros(self.n_hidden + 1) for _ in range(self.n_tasks)]

    def _activate(self, x):
        """Apply activation function to hidden units"""
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'linear':
            return x
        else:
            raise ValueError("Activation must be 'tanh' or 'linear'")

    def compute_hidden_activations(self, X):
        """Compute hidden unit activations with bias"""
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        H = self._activate(np.dot(X_bias, self.W.T))
        return np.hstack([H, np.ones((H.shape[0], 1))])

    def predict(self, X, task_idx):
        """Make predictions for a specific task"""
        H_bias = self.compute_hidden_activations(X)
        return np.dot(H_bias, self.A_map[task_idx])

    def compute_sufficient_statistics(self, X, y):
        """Compute sufficient statistics for a single task"""
        H_bias = self.compute_hidden_activations(X)
        return {
            'sum_hhT': np.dot(H_bias.T, H_bias),
            'sum_hy': np.dot(H_bias.T, y),
            'sum_yy': np.dot(y, y),
            'n_samples': X.shape[0]
        }

    def _normalize_data(self, X_list, y_list):
        """Normalize input data"""
        X_norm = [(X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8) for X in X_list]
        y_norm = [(y - np.mean(y)) / (np.std(y) + 1e-8) for y in y_list]
        return X_norm, y_norm