from copy import copy
from dataclasses import dataclass
from typing import Iterator
import numpy as np

# incapsulate the message


def m_norm(x, L):
    L = np.linalg.cholesky(L)
    y = np.linalg.solve(L, x)
    return np.dot(y, y)


@dataclass
class Item:
    """
    class to handle the items of coalescent_tree
    """
    mean: list
    cov: float = 0.
    t: float = 0.
    parent = None

    @staticmethod
    def coalesce(first, other, t,):
        """
        This method coalesces two Item objects. Works for brownian diffusion.
        For more see [2]

        [2] Y.W. Teh, H. Dauḿe III, and D. Roy. 
            Bayesian agglomerative clustering with coalescents. NIPS, 2007.
        """
        # messages coalesce
        # assert (self.cov + self.t - t) > 0
        # assert (other.cov + other.t - t) > 0

        l_inv = 1 / (first.cov + (first.t - t))
        r_inv = 1 / (other.cov + (other.t - t))
        new_cov = 1/(l_inv + r_inv)
        new_mean = (first.mean * l_inv + other.mean * r_inv) * new_cov

        new_t = t
        return Item(new_mean, new_cov, new_t)
    
    @staticmethod
    def optimal_t(first, other, cov, i, t_m1, n, dim):
        """
        computes optimal delta to coalesce for proposed pair of items
        """
        c_ni = ((n - i + 1) * (n-i)/2)

        a = 1/4 * 1/c_ni * \
            ((4 * c_ni * m_norm(first.mean - other.mean, cov) +
                dim*dim) ** 0.5 - dim)

        b = 1/2 * (first.cov + other.cov + first.t + other.t - 2 * t_m1)

        assert a - b > 0, f"t_m1: {t_m1}, a: {a}, b: {b}"
        return a - b


class CoalescentTree:
    def __init__(self, leaves: list[Item], cov, dim):
        self.leaves = leaves
        self.items, _ = self._greedy_rate_brownian(leaves,  cov, dim)

    def iterate_over(self) -> Iterator[Item]:
        """
        iterates over all elements in tree
        """
        for level in self.items:
            for node in level:
                yield node

    @staticmethod
    def select_candidates(x, cov, t_m1, n, dim):
        """
        Realise greedy selection of candidates to coalesce.
        For this consider all pairs of items and select minimal delta for coalesce

        More details in [2]
        """
        n_cur = len(x)
        i_items = n - n_cur

        l, r = None, None
        min_time = float("inf")

        for i in range(n_cur):
            for j in range(i + 1, n_cur):
                t_new = Item.optimal_t(x[i], x[j], cov, i_items, t_m1, n, dim)
                if t_new < min_time:
                    min_time, l, r = t_new, i, j
        return l, r, min_time

    def _greedy_rate_brownian(self, x: list[Item], cov, dim)\
            -> tuple[list[list[Item]], list[tuple[int, int]]]:
        """
        Realises greedy coalescent three construction
        More details in [2]

        :return: first argument is a constructed coalesce structure, 
            the second argument is a pointer to child elements
        """

        n = len(x)
        y = [x]
        coalesced_items = []
        t = 0
        for _ in range(n-1):
            coalecse_candidates = copy(y[-1])
            l, r, delta = self.select_candidates(coalecse_candidates,
                                                cov,
                                                t,
                                                n,
                                                dim)
            coalesced_items.append((l, r))
            t = t - delta
            new_item = Item.coalesce(coalecse_candidates[l],
                                    coalecse_candidates[r],
                                    t)

            coalecse_candidates[l].parent = new_item
            coalecse_candidates[r].parent = new_item

            coalecse_candidates[l] = new_item
            del coalecse_candidates[r]
            y.append(coalecse_candidates)
        return y, coalesced_items
    
