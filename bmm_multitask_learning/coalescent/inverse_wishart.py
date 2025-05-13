
import numpy as np


class InverseWishart:
    """
    class to handle methods of InverseWishart distribution
    """
    def __init__(self, d, degrees_of_freedom, init_cov=None, corr_mat=False):
        """
        :param d: dimension of the covariance matrix
        :param degrees_of_freedom: degrees of freedom of the
            inverse Wishart distribution
        :param init_cov: initial covariance matrix
        :param corr_mat: if True, the covariance matrix is a correlation
            matrix (i.e., it has ones on the diagonal)
        """
        self.d = d
        if init_cov is None:
            init_cov = np.eye(d)
        else:
            assert init_cov.shape == (d, d)

        self.cov = init_cov
        self.degrees_of_freedom = degrees_of_freedom
        self.corr_mat = corr_mat
    
    @staticmethod
    def cov2corr(cov):
        """
        Conver covariance matrox to correlation matrix
        """
        if len(cov.shape) == 0:
            cov = cov[None]

        std_devs = 1/np.sqrt(np.diag(cov))
        D = np.diag(std_devs)
        
        d = len(cov.shape)

        if d == 1:
            D = D[None]
            cov = cov[None]
        R = D @ cov @ D
        return R
        
    def update_posterior(self, samples):
        """
        :param samples: shape=(n x d)
        """
        assert len(samples.shape) == 2 and samples.shape[1] == self.d
        K = samples.shape[0]
        self.degrees_of_freedom += K

        S = samples.T @ samples  # d x d
        self.cov += S
        return
    
    def get_most_prob(self):
        """
        returns the mode of the inverse Wishart distribution
        """
        cov = self.cov
        cov = cov/(self.degrees_of_freedom + self.d + 1)
        
        if self.corr_mat:
            cov = InverseWishart.cov2corr(cov)
        return cov
        
