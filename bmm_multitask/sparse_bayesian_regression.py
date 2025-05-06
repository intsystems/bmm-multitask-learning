import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Dict, Any
from scipy.optimize import root
from scipy.special import kv, psi
import numpy as np


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
        G = len(self.group_indices)
        # Общие (prior) параметры для всех групп
        self.omega_prior = torch.tensor(1.0, device=self.device, requires_grad=False)
        self.chi_prior = torch.tensor(1.0, device=self.device, requires_grad=False)
        self.phi_prior = torch.tensor(1.0, device=self.device, requires_grad=False)
        self.nu_prior = torch.tensor(1.0, device=self.device, requires_grad=False) # гиперпараметр для v_i
        # Постериорные параметры для каждой группы
        self.omega_post = [torch.tensor(1.0, device=self.device, requires_grad=False) for _ in range(G)]
        self.chi_post = [torch.tensor(1.0, device=self.device, requires_grad=False) for _ in range(G)]
        self.phi_post = [torch.tensor(1.0, device=self.device, requires_grad=False) for _ in range(G)]
        self.nu_post = [torch.tensor(1.0, device=self.device, requires_grad=False) for _ in range(G)]
        self.tau = torch.tensor(1.0, device=self.device, requires_grad=False)  # дисперсия шума
        self.sigma2 = torch.tensor(1.0, device=self.device, requires_grad=False)  # дисперсия шума
        self.K = 2  # ранг латентного пространства, можно параметризовать
        self.P = self.model(torch.zeros(1, self.model.in_features, device=self.device)).shape[-1] # количество выходов
        self.D = self.model.in_features # количество входов
        self.Omega_inv_g = [torch.eye(len(idxs), device=self.device) for idxs in self.group_indices]  # [D_g]
        self.gammas = [1.0 for _ in range(G)]
        # Параметры постериорного распределения для W, Z, V
        # W: матрично-нормальное (M_W, Omega_W, S_W)
        self.M_W = torch.randn(self.P, self.D, device=self.device) / np.sqrt(self.P * self.D)
        self.Omega_W = torch.eye(self.D, device=self.device)
        self.S_W = torch.eye(self.P, device=self.device)
        # Z: список по группам (M_Z[g], Omega_Z[g], S_Z[g])
        self.M_Z = [torch.randn(self.K, len(idxs), device=self.device) / np.sqrt(self.K *  len(idxs)) for idxs in self.group_indices]
        self.Omega_Z = [torch.eye(len(idxs), device=self.device) for idxs in self.group_indices]
        self.S_Z = [torch.eye(self.K, device=self.device) for _ in self.group_indices]
        # V: матрично-нормальное (M_V, Omega_V, S_V)
        self.M_V = torch.zeros(self.P, self.K, device=self.device)
        self.Omega_V = torch.eye(self.K, device=self.device)
        self.S_V = torch.eye(self.P, device=self.device)

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

    @staticmethod
    def mean_inv_gig(omega: float, chi: float, phi: float) -> float:
        """
        Математическое ожидание обратной величины ⟨1/x⟩ для GIG(omega, chi, phi)
        ⟨1/x⟩ = sqrt(chi/phi) * R_{omega-1}(sqrt(chi*phi))
        где R_{omega-1}(z) = K_{omega}(z) / K_{omega-1}(z)
        """
        z = (chi * phi) ** 0.5
        K_omega = kv(omega, z)
        K_omega_m1 = kv(omega - 1, z)
        R_omega = K_omega_m1 / K_omega if K_omega != 0 else 0.0
        return (phi / chi) ** 0.5 * R_omega

    @staticmethod
    def mean_log_gig(omega: float, chi: float, phi: float) -> float:
        """
        Математическое ожидание логарифма ⟨log(x)⟩ для GIG(omega, chi, phi)
        """
        z = (chi * phi) ** 0.5
        return 0.5 * np.log(chi / phi) + (SparseBayesianRegression.d_log_bessel_k(omega, z))

    @staticmethod
    def d_log_bessel_k(omega, z):
        # Производная по omega от log K_omega(z)
        eps = 1e-5
        return (np.log(kv(omega + eps, z)) - np.log(kv(omega - eps, z))) / (2 * eps)

    def update_gig_hyperparams(self, group_idx, mean_gamma, mean_inv_gamma, mean_log_gamma):
        # Численно решает систему для omega, chi, phi для одной группы
        Q = 1  # для одной группы, если групп больше, можно обобщить
        def equations(params):
            omega, chi, phi = params
            z = np.sqrt(chi * phi)
            K_omega = kv(omega, z)
            d_logK = self.d_log_bessel_k(omega, z)
            R_omega = kv(omega + 1, z) / K_omega if K_omega != 0 else 0.0
            eq1 = Q * np.log(np.sqrt(phi / chi)) - Q * d_logK - Q * mean_log_gamma
            eq2 = (Q * omega) / chi - (Q / 2) * np.sqrt(phi / chi) * R_omega + 0.5 * mean_inv_gamma
            eq3 = (Q / np.sqrt(chi * phi)) * R_omega - mean_gamma
            return [eq1, eq2, eq3]
        # Начальные значения
        omega0 = float(self.omega_post[group_idx].cpu().numpy())
        chi0 = float(self.chi_post[group_idx].cpu().numpy())
        phi0 = float(self.phi_post[group_idx].cpu().numpy())
        sol = root(equations, [omega0, chi0, phi0], method='hybr')
        if sol.success:
            self.omega_post[group_idx] = torch.tensor(sol.x[0], device=self.device)
            self.chi_post[group_idx] = torch.tensor(sol.x[1], device=self.device)
            self.phi_post[group_idx] = torch.tensor(sol.x[2], device=self.device)

    def update_gig_prior(self, mean_gammas, mean_inv_gammas, mean_log_gammas):
        # Численно решает систему для omega, chi, phi для общего прайора (по средним по группам)
        Q = len(mean_gammas)
        sum_gamma = torch.sum(torch.tensor(mean_gammas))
        sum_inv_gamma = torch.sum(torch.tensor(mean_inv_gammas))
        sum_log_gamma = torch.sum(torch.tensor(mean_log_gammas))
        def equations(params):
            omega, chi, phi = params
            z = np.sqrt(chi * phi)
            K_omega = kv(omega, z)
            d_logK = self.d_log_bessel_k(omega, z)
            R_omega = kv(omega + 1, z) / K_omega if K_omega != 0 else 0.0
            eq1 = Q * np.log(np.sqrt(phi / chi)) - Q * d_logK * sum_log_gamma
            eq2 = (Q * omega) / chi - (Q / 2) * np.sqrt(phi / chi) * R_omega + 0.5 * sum_inv_gamma
            eq3 = Q * np.sqrt(chi/ phi) * R_omega - sum_gamma
            return [eq1, eq2, eq3]
        omega0 = float(self.omega_prior.cpu().numpy())
        chi0 = float(self.chi_prior.cpu().numpy())
        phi0 = float(self.phi_prior.cpu().numpy())
        sol = root(equations, [omega0, chi0, phi0], method='hybr')
        if sol.success:
            self.omega_prior = torch.tensor(sol.x[0], device=self.device)
            self.chi_prior = torch.tensor(sol.x[1], device=self.device)
            self.phi_prior = torch.tensor(sol.x[2], device=self.device)

    def compute_moments_W(self, M_W, Omega_W, S_W):
        # Момент: E[W W^T] = M_W M_W^T + tr(S_W) * Omega_W
        return M_W @ M_W.t() + torch.trace(S_W) * Omega_W

    def compute_moments_Z(self, M_Z, Omega_Z, S_Z):
        # Момент: E[Z Z^T] = M_Z M_Z^T + tr(S_Z) * Omega_Z
        return M_Z @ M_Z.t() + torch.trace(S_Z) * Omega_Z

    def compute_moments_VVT(self, M_V, Omega_V, S_V):
        # Момент: E[V V^T] = M_V M_V^T + tr(S_V) * Omega_V
        return M_V @ M_V.t() + torch.trace(Omega_V) * S_V
    def compute_moments_VTV(self, M_V, Omega_V, S_V):
        # Момент: E[V V^T] = M_V M_V^T + tr(S_V) * Omega_V
        return M_V.T @ M_V + torch.trace(S_V) * Omega_V

    def e_step(self, X: Tensor, Y: Tensor) -> Dict[str, Any]:
        """
        E-шаг: вычисляет апостериорные параметры (среднее и ковариацию) для W, Z, V
        Формулы из Supplement C (см. скриншот)
        """
        
        D, N = X.shape
        tau = self.tau
        sigma2 = self.sigma2  # можно вынести в параметры класса

        # --- E-step: обновление параметров постериора для W ---
        # <Gamma> и <Omega^{-1}> по блокам
        G = len(self.group_indices)
        D = self.D
        P = self.P
        K = self.K
        tau = self.tau
        sigma2 = self.sigma2
        # 1. Собираем <Gamma> и <Omega^{-1}> как блочные диагональные матрицы
        mean_gammas = []
        mean_inv_gammas = []
        mean_log_gammas = []
        Gamma = torch.zeros(D, D, device=self.device)
        Omega_inv = torch.zeros(D, D, device=self.device)
        for g, idxs in enumerate(self.group_indices):
            omega = float(self.omega_post[g].cpu().numpy())
            chi = float(self.chi_post[g].cpu().numpy())
            phi = float(self.phi_post[g].cpu().numpy())
            mg = self.mean_gig(omega, chi, phi)
            mig = self.mean_inv_gig(omega, chi, phi)
            mlg = self.mean_log_gig(omega, chi, phi)
            mean_gammas.append(mg)
            mean_inv_gammas.append(mig)
            mean_log_gammas.append(mlg)
            print(Gamma[idxs, :][:, idxs])
            Gamma[idxs, :][:, idxs] = mg * torch.eye(len(idxs), device=self.device)
            Omega_inv[idxs, :][:, idxs] = self.Omega_inv_g[g]
        # 2. W: матрично-нормальное
        # Omega_W = (tau^{-1} <Omega^{-1}><Gamma> + sigma^{-2} X X^T)^{-1}
        Omega_W_inv = (1.0 / tau) * Omega_inv @ Gamma + (1.0 / sigma2) * (X @ X.t())
        Omega_W = torch.linalg.inv(Omega_W_inv)
        # M_W = (tau^{-1} V Z <Omega^{-1}><Gamma> + sigma^{-2} Y X^T) Omega_W
        V = self.M_V
        Z = torch.cat(self.M_Z, dim=1)  # [K, D]
        M_W = ((1.0 / tau) * V @ Z @ Omega_inv @ Gamma + (1.0 / sigma2) * Y @ X.t()) @ Omega_W
        S_W = torch.eye(P, device=self.device)
        # 3. Z: по группам
        M_Z = []
        Omega_Z = []
        S_Z = []
        for g, idxs in enumerate(self.group_indices):
            Dg = len(idxs)
            Wg = M_W[:, idxs]  # [Dg, P]
            V = self.M_V
            # Используем момент ⟨V^T V⟩ для S_Zi
            moment_V = self.compute_moments_VTV(self.M_V, self.Omega_V, self.S_V)
            S_Zi = torch.linalg.inv((1.0 / tau) * moment_V + torch.eye(K, device=self.device))
            # M_Zi = tau^{-1} S_Zi V^T W_i
            M_Zi = (1.0 / tau) * S_Zi @ V.t() @ Wg
            # Omega_Zi = <gamma_i>^{-1} <Omega_i^{-1}>^{-1}
            Omega_i_inv = self.Omega_inv_g[g]
            Omega_Zi = (1.0 / mean_gammas[g]) * torch.linalg.inv(Omega_i_inv)
            M_Z.append(M_Zi)
            Omega_Z.append(Omega_Zi)
            S_Z.append(S_Zi)
        # 4. V
        # Omega_V = (sum_i <gamma_i> E[Z_i Omega_i^{-1} Z_i^T] + I_K)^{-1}
        Omega_V_inv = torch.zeros(K, K, device=self.device)
        for g, idxs in enumerate(self.group_indices):
            # E[Z_i Omega_i^{-1} Z_i^T] = Omega_i^{-1} tr(S_Zi) + M_Zi M_Zi^T
            Omega_i_inv = self.Omega_inv_g[g]
            M_Zi = M_Z[g]
            S_Zi = S_Z[g]
            moment_Z =  Omega_Zi * torch.sum(S_Zi @ Omega_i_inv) + M_Zi @ Omega_i_inv @ M_Zi.t()
            Omega_V_inv += mean_gammas[g] * moment_Z
        Omega_V_inv += torch.eye(K, device=self.device)
        Omega_V = torch.linalg.inv(Omega_V_inv)
        # M_V = M_W @ Omega_inv @ Gamma @ Z.T @ Omega_V
        M_V = M_W @ Omega_inv @ Gamma @ Z.T @ Omega_V
        S_V = tau * torch.eye(P, device=self.device)
        # Сохраняем параметры постериорного распределения после расчёта
        self.M_W = M_W
        self.Omega_W = Omega_W
        self.S_W = S_W
        self.M_Z = M_Z
        self.Omega_Z = Omega_Z
        self.S_Z = S_Z
        self.M_V = M_V
        self.Omega_V = Omega_V
        self.S_V = S_V
        # --- Обновление параметров постериора обратного Wishart для каждой группы ---
        self.Lambda = []

        lambda_reg = 1.0  # при необходимости вынести в параметры класса
        for g, idxs in enumerate(self.group_indices):
            Dg = len(idxs)
            P = self.P
            K = self.K
            # W_i, Z_i для группы
            Wg = self.M_W[:, idxs]  # [P, Dg]
            V = self.M_V #[P, K]
            Zg = self.M_Z[g]  # [Dg, K]
            # Остаток: W_i - V Z_i
            resid = Wg - V @ Zg.t()  # [P, Dg]
            # Момент: <(W_i - VZ_i)(W_i - VZ_i)^T>
            # E[W_i W_i^T]
            Omega_W_g = Omega_W[idxs, :][:, idxs]
            D_W = torch.trace(S_W) * Omega_W_g
            # E[Z_i Z_i^T]
            E_ZZ = Zg @ Zg.t() + torch.trace(self.S_Z[g]) * self.Omega_Z[g]
            # E[W_i Z_i^T]
            E_WZ = Wg @ Zg.t()
            # E[Z_i W_i^T]
            E_ZW = Zg @ Wg.t()
            # V
            V = self.M_V
            moment_resid_mean_part = (Wg - V @ Zg.t()).T @ (Wg - V @ Zg.t())
            
            moment_resid_disp_part = D_W + Omega_Zi * torch.sum(S_Zi * Omega_V) * torch.trace(S_V) + M_Zi.T @ Omega_V @M_Zi * torch.trace(S_V)+\
            Omega_Zi * torch.sum(S_Zi * (M_Zi.T @ M_Zi))
            moment_resid = moment_resid_mean_part + moment_resid_disp_part
            Lambda_i = (1.0 / tau) * mean_gammas[g] * moment_resid + mean_gammas[g] * moment_Z + lambda_reg * torch.eye(Dg, device=self.device)
            self.Lambda.append(Lambda_i)
        # Подсчет v_i для каждой группы (по формуле v_i = v + P + K)
        self.nu_post = []
        for g, idxs in enumerate(self.group_indices):
            nu_i = float(self.nu_prior) + self.P + self.K
            self.nu_post.append(nu_i)
        # Подсчет <Omega_i^{-1}> для каждой группы
        
        self.Omega_inv_g = []
        for g, idxs in enumerate(self.group_indices):
            Dg = len(idxs)
            nu_i = self.nu_post[g]
            Lambda_i = self.Lambda[g]
            Omega_inv_i = (Dg + nu_i - 1) * torch.linalg.inv(Lambda_i)
            self.Omega_inv_g.append(Omega_inv_i)
        # Обновление параметров постериорного GIG для каждой группы
        for g, idxs in enumerate(self.group_indices):
            Dg = len(idxs)
            # omega_post
            self.omega_post[g] = self.omega_prior + 0.5 * (self.P + self.K) * Dg
            # chi_post
            self.chi_post[g] = self.chi_prior
            # phi_post
            Omega_i_inv = self.Omega_inv_g[g]
            Wg = self.M_W[:, idxs]  # [P, Dg]
            V = self.M_V # [P, K]
            Zg = self.M_Z[g]  # [Dg, K]
            resid = Wg - V @ Zg.t()  # [P, Dg]
            Omega_W_g = Omega_W[idxs, :][:, idxs]
            # Момент: <(W_i - VZ_i) Omega_i^{-1} (W_i - VZ_i)^T>
            Omega_inv_i = self.Omega_inv_g[g]
            moment_resid_mean_part = resid @ Omega_i_inv @ resid.t()
            moment_resid_W_disp_part = S_W * torch.sum(Omega_W_g * Omega_inv_i)
            moment_resid_VZ_disp_part = S_V * torch.sum(Omega_V * S_Zi) * torch.sum(Omega_Zi* Omega_i_inv) + M_V @ S_Zi @M_V.T * torch.sum(Omega_Zi* Omega_i_inv)+\
            S_V * torch.sum(Omega_V * (M_Zi @ Omega_inv_i @M_Zi.T))
            moment_resid = moment_resid_mean_part + moment_resid_W_disp_part + moment_resid_VZ_disp_part
            tr_resid = torch.trace(moment_resid)
            # Момент: <Z_i Omega_i^{-1} Z_i^T>
            moment_Z = Zg @ Omega_i_inv @ Zg.t() + torch.trace(self.S_Z[g]) * Omega_i_inv
            tr_Z = torch.trace(moment_Z)
            self.phi_post[g] = self.phi_prior + (1.0 / self.tau) * tr_resid + tr_Z
        mean_gammas = []
        mean_inv_gammas = []
        mean_log_gammas = []
        for g, idxs in enumerate(self.group_indices):
            omega = float(self.omega_post[g].cpu().numpy())
            chi = float(self.chi_post[g].cpu().numpy())
            phi = float(self.phi_post[g].cpu().numpy())
            mg = self.mean_gig(omega, chi, phi)
            mig = self.mean_inv_gig(omega, chi, phi)
            mlg = self.mean_log_gig(omega, chi, phi)
            mean_gammas.append(mg)
            mean_inv_gammas.append(mig)
            mean_log_gammas.append(mlg)
        return {
            'mean_gammas': mean_gammas,
            'mean_inv_gammas': mean_inv_gammas,
            'mean_log_gammas': mean_log_gammas,
            "M_W": M_W, "Omega_W": Omega_W, "S_W": S_W,
            "M_Z": M_Z, "Omega_Z": Omega_Z, "S_Z": S_Z,
            "M_V": M_V, "Omega_V": Omega_V, "S_V": S_V
        }

    def m_step(self, post: Dict[str, Any]):
        """
        M-шаг: обновление гиперпараметров (раздел 4.2)
        """
        # Теперь только обновление общего прайора
        self.update_gig_prior(post['mean_gammas'], post['mean_inv_gammas'], post['mean_log_gammas'])
        # Обновление tau (дисперсия шума)
        # ... (см. 4.2, зависит от задачи)

    def fit(self, X: Tensor, Y: Tensor, num_iter: int = 10):
        X = X.T
        for _ in range(num_iter):
            post = self.e_step(X, Y)
            self.m_step(post)
            self._set_flat_params(post["M_W"])

    def predict(self, X: Tensor) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(X)
