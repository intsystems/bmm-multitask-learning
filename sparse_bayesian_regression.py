import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Dict, Any
from scipy.special import kv


class SparseBayesianRegression:
    def __init__(self, model: nn.Module, group_indices: List[List[int]],
                 device: Optional[str] = None):
        """
        model: torch.nn.Module (линейная или любая torch-модель)
        group_indices: список списков индексов параметров, соответствующих группам
        device: cpu/cuda
        """
        self.model = model
        self.group_indices = group_indices
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._init_hyperparams()

    def _init_hyperparams(self):
        # Инициализация гиперпараметров прайора для каждой группы
        self.omega= torch.tensor(1.0, device=self.device, requires_grad=False)
        self.chi = torch.tensor(1.0, device=self.device, requires_grad=False)
        self.phi = torch.tensor(1.0, device=self.device, requires_grad=False)
        self.nu = torch.tensor(1.0, device=self.device, requires_grad=False)
        self.tau = torch.tensor(1.0, device=self.device, requires_grad=False)  # дисперсия шума
        self.sigma2 = torch.tensor(1.0, device=self.device, requires_grad=False)  # дисперсия шума
        # Инициализация V и Z для честного учета ковариации задач
        self.K = 2  # ранг латентного пространства, можно параметризовать
        self.P = self.model(X=torch.zeros(1, self.model.in_features, device=self.device)).shape[-1]
        self.V = torch.randn(self.P, self.K, device=self.device)  # [P, K]
        self.Z = [torch.randn(len(idxs), self.K, device=self.device) for idxs in self.group_indices]  # [D_g, K] для каждой группы
        self.W = self._get_flat_params()
        self.D = self.model.in_features
        self.Omega_inv_g = [torch.eye(self.D, device=self.device) for _ in self.group_indices]  # [D_g]
    def _get_flat_params(self) -> Tensor:
        # Вытягивает параметры модели в один вектор
        return torch.cat([p.view(-1) for p in self.model.parameters()])

    def _set_flat_params(self, flat_params: Tensor):
        # Устанавливает параметры модели из вектора
        pointer = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[pointer:pointer+numel].view_as(p))
            pointer += numel
    @staticmethod
    def mean_gig(omega: float, chi: float, phi: float) -> float:
        """
        Математическое ожидание ⟨x⟩ для GIG(omega, chi, phi)
        ⟨x⟩ = sqrt(chi/phi) * R_omega(sqrt(chi*phi))
        где R_omega(z) = K_{omega+1}(z) / K_{omega}(z)
        """
        z = (chi * phi) ** 0.5
        K_omega = kv(omega, z)
        K_omega_p1 = kv(omega + 1, z)
        R_omega = K_omega_p1 / K_omega if K_omega != 0 else 0.0
        return (chi / phi) ** 0.5 * R_omega
    def e_step(self, X: Tensor, Y: Tensor) -> Dict[str, Any]:
        """
        E-шаг: вычисляет апостериорные параметры (среднее и ковариацию) для W, Z, V
        Формулы из Supplement C (см. скриншот)
        """
        D, N = X.shape
        tau = self.tau
        sigma2 = self.sigma2  # можно вынести в параметры класса
        nu = self.nu
        # <V>, <Z>
        Z = self.Z  # [D, K] если одна группа
        V = self.V     # [P, K]
        # --- W ---
        # prior_prec: [D]
        D_g = []
        # <Gamma>
        Gammas = []
        nus = []
        W_g = []
        for g, idxs in enumerate(self.group_indices):
            Gammas.append(self.mean_gig(self.omega[g], self.chi[g], self.phi[g]))
            D_g.append(len(idxs))
            nus.append(self.nu[g] + self.P + self.K)
            W_g.append(self.W[idxs])  # [Dg, P]
        Gamma = torch.zeros(D, D, device=self.device)
        for g, idxs in enumerate(self.group_indices):
            Gamma[idxs, idxs] = Gammas[g] * torch.eye(D_g[g], device=self.device)
        
        # <Omega^{-1}> = I_D (можно параметризовать)
        Omega_inv_g = self.Omega_inv_g
        Omega_inv = torch.zeros(D, D, device=self.device)
        for g, idxs in enumerate(self.group_indices):
            Omega_inv[idxs, idxs] = Omega_inv_g[g]
        # Omega_W = (tau^{-1} <Omega^{-1}><Gamma> + sigma^{-2} X X^T)^{-1}
        
        Omega_W_inv = (1.0 / tau) * Omega_inv @ Gamma + (1.0 / sigma2) * (X @ X.t())
        Omega_W = torch.linalg.inv(Omega_W_inv)
        # M_W = (tau^{-1} <V><Z><Omega^{-1}><Gamma> + sigma^{-2} Y X^T) Omega_W
        M_W = ((1.0 / tau) * V @ Z @ Omega_inv @ Gamma + (1.0 / sigma2) * Y @ X.t()) @ Omega_W
        S_W = torch.eye(V.shape[0], device=self.device)  # S_W = I_P

        # --- Z_i (по группам) ---
        M_Z = []
        Omega_Z = []
        S_Z = []
        for g, idxs in enumerate(self.group_indices):
            Dg = len(idxs)
            Zg = self.Z[g]  # [Dg, K]
            Wg = M_W[:, idxs].t()  # [Dg, P] -> [Dg, P]
            # S_Zi = (tau^{-1} V^T V + I_K)^{-1}
            S_Zi = torch.linalg.inv((1.0 / tau) * V.t() @ V + torch.eye(self.K, device=self.device))
            # M_Zi = tau^{-1} S_Zi <V^T> <W_i>
            M_Zi = (1.0 / tau) * S_Zi @ V.t() @ Wg
            # <Omega_i^{-1}> = I_Dg (можно параметризовать)
            Omega_i_inv = Omega_inv_g[g]
            # Omega_Zi = <gamma_i>^{-1} (<Omega_i^{-1}>)^{-1}
            Omega_Zi = (1.0 / Gammas[g]) * torch.linalg.inv(Omega_i_inv)
            M_Z.append(M_Zi)
            Omega_Z.append(Omega_Zi)
            S_Z.append(S_Zi)

        # --- V ---
        # Omega_V = (sum_i <gamma_i> Z_i <Omega_i^{-1}> Z_i^T + I_K)^{-1}
        Omega_V_inv = torch.zeros(self.K, self.K, device=self.device)
        for g, idxs in enumerate(self.group_indices):
            Zg = self.Z[g]  # [Dg, K]
            Dg = Zg.shape[0]
            Omega_i_inv = torch.eye(Dg, device=self.device)
            Omega_V_inv += self.gammas[g] * Zg.t() @ Omega_i_inv @ Zg
        Omega_V_inv += torch.eye(self.K, device=self.device)
        Omega_V = torch.linalg.inv(Omega_V_inv)
        # M_V = <W> <Omega^{-1}> <Gamma> <Z^T> Omega_V
        # <W>: [P, D], <Omega^{-1}>: [D, D], <Gamma>: [D, D], <Z^T>: [D, K]
        # Для одной группы:
        Z = self.Z[0]
        Gamma = torch.diag(prior_prec)
        M_V = M_W @ Omega_inv @ Gamma @ Z @ Omega_V
        S_V = tau * torch.eye(V.shape[0], device=self.device)

        return {
            "M_W": M_W, "Omega_W": Omega_W, "S_W": S_W,
            "M_Z": M_Z, "Omega_Z": Omega_Z, "S_Z": S_Z,
            "M_V": M_V, "Omega_V": Omega_V, "S_V": S_V
        }

    def m_step(self, post: Dict[str, Any]):
        """
        M-шаг: обновление гиперпараметров (раздел 4.2)
        """
        mu = post["mu"]
        Sigma = post["Sigma"]
        for g, idxs in enumerate(self.group_indices):
            mu_g = mu[idxs]
            Sigma_gg = Sigma[idxs][:, idxs]
            # Обновление λ_g и γ_g (см. 4.2)
            E_w2 = mu_g @ mu_g + torch.trace(Sigma_gg)
            self.lambdas[g] = (len(idxs) + 2 * 1e-6) / (E_w2 + 2 * 1e-6)  # регуляризация
            self.gammas[g] = 1.0  # если требуется, можно обновлять по формуле из 4.2
        # Обновление tau (дисперсия шума)
        # ... (см. 4.2, зависит от задачи)

    def fit(self, X: Tensor, Y: Tensor, num_iter: int = 10):
        for _ in range(num_iter):
            post = self.e_step(X, Y)
            self.m_step(post)
            self._set_flat_params(post["mu"])

    def predict(self, X: Tensor) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(X)
