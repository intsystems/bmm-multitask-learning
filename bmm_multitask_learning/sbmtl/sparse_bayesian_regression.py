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

    def _init_hyperparams(self) -> None:
        """
        Инициализирует гиперпараметры модели, включая параметры прайора и постериора.
        """
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
        """
        Вытягивает параметры модели в один вектор.
        Возвращает:
            Tensor: Вектор параметров модели.
        """
        # Вытягивает параметры модели в один вектор
        return torch.cat([p.view(-1) for p in self.model.parameters()])

    def _set_flat_params(self, flat_params: Tensor) -> None:
        """
        Устанавливает параметры модели из вектора.
        Аргументы:
            flat_params (Tensor): Вектор параметров модели.
        """
        # Устанавливает параметры модели из вектора
        pointer = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[pointer:pointer+numel].view_as(p))
            pointer += numel
    @staticmethod
    def mean_gig(omega: float, chi: float, phi: float) -> float:
        """
        Вычисляет математическое ожидание ⟨x⟩ для GIG(omega, chi, phi).
        ⟨x⟩ = sqrt(chi/phi) * R_omega(sqrt(chi*phi))
        где R_omega(z) = K_{omega+1}(z) / K_{omega}(z)
        Аргументы:
            omega (float): Параметр omega распределения GIG.
            chi (float): Параметр chi распределения GIG.
            phi (float): Параметр phi распределения GIG.
        Возвращает:
            float: Математическое ожидание ⟨x⟩.
        """
        z = (chi * phi) ** 0.5
        K_omega = kv(omega, z)
        K_omega_p1 = kv(omega + 1, z)
        R_omega = K_omega_p1 / K_omega if K_omega != 0 else 0.0
        return (chi / phi) ** 0.5 * R_omega

    @staticmethod
    def mean_inv_gig(omega: float, chi: float, phi: float) -> float:
        """
        Вычисляет математическое ожидание обратной величины ⟨1/x⟩ для GIG(omega, chi, phi).
        ⟨1/x⟩ = sqrt(chi/phi) * R_{omega-1}(sqrt(chi*phi))
        где R_{omega-1}(z) = K_{omega}(z) / K_{omega-1}(z)
        Аргументы:
            omega (float): Параметр omega распределения GIG.
            chi (float): Параметр chi распределения GIG.
            phi (float): Параметр phi распределения GIG.
        Возвращает:
            float: Математическое ожидание ⟨1/x⟩.
        """
        z = (chi * phi) ** 0.5
        K_omega = kv(omega, z)
        K_omega_m1 = kv(omega - 1, z)
        R_omega = K_omega_m1 / K_omega if K_omega != 0 else 0.0
        return (phi / chi) ** 0.5 * R_omega

    @staticmethod
    def mean_log_gig(omega: float, chi: float, phi: float) -> float:
        """
        Вычисляет математическое ожидание логарифма ⟨log(x)⟩ для GIG(omega, chi, phi).
        Аргументы:
            omega (float): Параметр omega распределения GIG.
            chi (float): Параметр chi распределения GIG.
            phi (float): Параметр phi распределения GIG.
        Возвращает:
            float: Математическое ожидание ⟨log(x)⟩.
        """
        z = (chi * phi) ** 0.5
        return 0.5 * np.log(chi / phi) + (SparseBayesianRegression.d_log_bessel_k(omega, z))

    @staticmethod
    def d_log_bessel_k(omega: float, z: float) -> float:
        """
        Вычисляет производную по omega от log K_omega(z).
        Аргументы:
            omega (float): Параметр omega.
            z (float): Параметр z.
        Возвращает:
            float: Значение производной.
        """
        # Производная по omega от log K_omega(z)
        eps = 1e-5
        return (np.log(kv(omega + eps, z)) - np.log(kv(omega - eps, z))) / (2 * eps)

    def update_gig_hyperparams(self, group_idx: int, mean_gamma: float, mean_inv_gamma: float, mean_log_gamma: float) -> None:
        """
        Обновляет гиперпараметры GIG (omega, chi, phi) для одной группы.
        Аргументы:
            group_idx (int): Индекс группы.
            mean_gamma (float): Математическое ожидание ⟨gamma⟩.
            mean_inv_gamma (float): Математическое ожидание ⟨1/gamma⟩.
            mean_log_gamma (float): Математическое ожидание ⟨log(gamma)⟩.
        """
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

    def update_gig_prior(self, mean_gammas: List[float], mean_inv_gammas: List[float], mean_log_gammas: List[float]) -> None:
        """
        Обновляет гиперпараметры GIG (omega, chi, phi) для общего прайора.
        Аргументы:
            mean_gammas (List[float]): Список математических ожиданий ⟨gamma⟩ для всех групп.
            mean_inv_gammas (List[float]): Список математических ожиданий ⟨1/gamma⟩ для всех групп.
            mean_log_gammas (List[float]): Список математических ожиданий ⟨log(gamma)⟩ для всех групп.
        """
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

    def compute_moments_W(self, M_W: Tensor, Omega_W: Tensor, S_W: Tensor) -> Tensor:
        """
        Вычисляет момент ⟨W W^T⟩ для матрицы W.
        Аргументы:
            M_W (Tensor): Матрица средних значений W.
            Omega_W (Tensor): Ковариационная матрица по строкам W.
            S_W (Tensor): Ковариационная матрица по столбцам W.
        Возвращает:
            Tensor: Момент ⟨W W^T⟩.
        """
        # Момент: E[W W^T] = M_W M_W^T + tr(S_W) * Omega_W
        return M_W @ M_W.t() + torch.trace(S_W) * Omega_W

    def compute_moments_Z(self, M_Z: Tensor, Omega_Z: Tensor, S_Z: Tensor) -> Tensor:
        """
        Вычисляет момент ⟨Z Z^T⟩ для матрицы Z.
        Аргументы:
            M_Z (Tensor): Матрица средних значений Z.
            Omega_Z (Tensor): Ковариационная матрица по строкам Z.
            S_Z (Tensor): Ковариационная матрица по столбцам Z.
        Возвращает:
            Tensor: Момент ⟨Z Z^T⟩.
        """
        # Момент: E[Z Z^T] = M_Z M_Z^T + tr(S_Z) * Omega_Z
        return M_Z @ M_Z.t() + torch.trace(S_Z) * Omega_Z

    def compute_moments_VVT(self, M_V: Tensor, Omega_V: Tensor, S_V: Tensor) -> Tensor:
        """
        Вычисляет момент ⟨V V^T⟩ для матрицы V.
        Аргументы:
            M_V (Tensor): Матрица средних значений V.
            Omega_V (Tensor): Ковариационная матрица по строкам V.
            S_V (Tensor): Ковариационная матрица по столбцам V.
        Возвращает:
            Tensor: Момент ⟨V V^T⟩.
        """
        # Момент: E[V V^T] = M_V M_V^T + tr(S_V) * Omega_V
        return M_V @ M_V.t() + torch.trace(Omega_V) * S_V
    def compute_moments_VTV(self, M_V: Tensor, Omega_V: Tensor, S_V: Tensor) -> Tensor:
        """
        Вычисляет момент ⟨V^T V⟩ для матрицы V.
        Аргументы:
            M_V (Tensor): Матрица средних значений V.
            Omega_V (Tensor): Ковариационная матрица по строкам V.
            S_V (Tensor): Ковариационная матрица по столбцам V.
        Возвращает:
            Tensor: Момент ⟨V^T V⟩.
        """
        # Момент: E[V V^T] = M_V M_V^T + tr(S_V) * Omega_V
        return M_V.T @ M_V + torch.trace(S_V) * Omega_V

    def e_step(self, X: Tensor, Y: Tensor) -> Dict[str, Any]:
        """
        E-шаг: координирует обновление всех параметров постериорных распределений и моментов.
        Аргументы:
            X (Tensor): матрица признаков (D, N)
            Y (Tensor): матрица откликов (P, N)
        Возвращает: dict[str, Any] — словарь с основными статистиками и параметрами для M-шагa.
        """
        group_moments = self._compute_group_moments()
        self._update_posterior_matrices(X, Y, group_moments)
        self._update_posterior_wishart(group_moments)
        self._update_posterior_gig(group_moments)
        return {
            'mean_gammas': group_moments['mean_gammas'],
            'mean_inv_gammas': group_moments['mean_inv_gammas'],
            'mean_log_gammas': group_moments['mean_log_gammas'],
            "M_W": self.M_W, "Omega_W": self.Omega_W, "S_W": self.S_W,
            "M_Z": self.M_Z, "Omega_Z": self.Omega_Z, "S_Z": self.S_Z,
            "M_V": self.M_V, "Omega_V": self.Omega_V, "S_V": self.S_V
        }

    def _compute_group_moments(self) -> Dict[str, Any]:
        """
        Вычисляет моменты (средние значения) по всем группам, а также блочные матрицы Gamma и Omega_inv.
        Возвращает: dict[str, Any] — словарь с этими величинами.
        """
        G = len(self.group_indices)
        D = self.D
        mean_gammas, mean_inv_gammas, mean_log_gammas = [], [], []
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
            Gamma[idxs, :][:, idxs] = mg * torch.eye(len(idxs), device=self.device)
            Omega_inv[idxs, :][:, idxs] = self.Omega_inv_g[g]
        return {
            'mean_gammas': mean_gammas,
            'mean_inv_gammas': mean_inv_gammas,
            'mean_log_gammas': mean_log_gammas,
            'Gamma': Gamma,
            'Omega_inv': Omega_inv
        }

    def _update_posterior_matrices(self, X: Tensor, Y: Tensor, group_moments: Dict[str, Any]) -> None:
        """
        Обновляет параметры постериорных матрично-нормальных распределений W, Z, V на основе текущих моментов и данных.
        Аргументы:
            X (Tensor): матрица признаков
            Y (Tensor): матрица откликов
            group_moments (dict): словарь с моментами и блочными матрицами
        Возвращает: None
        """
        Gamma = group_moments['Gamma']
        Omega_inv = group_moments['Omega_inv']
        tau = self.tau
        sigma2 = self.sigma2
        # Обновление W
        Omega_W_inv = (1.0 / tau) * Omega_inv @ Gamma + (1.0 / sigma2) * (X @ X.t())
        self.Omega_W = torch.linalg.inv(Omega_W_inv)
        Z = torch.cat(self.M_Z, dim=1)  # [K, D]
        self.M_W = ((1.0 / tau) * self.M_V @ Z @ Omega_inv @ Gamma + (1.0 / sigma2) * Y @ X.t()) @ self.Omega_W
        self.S_W = torch.eye(self.P, device=self.device)
        # Обновление Z
        self.M_Z, self.Omega_Z, self.S_Z = [], [], []
        for g, idxs in enumerate(self.group_indices):
            Dg = len(idxs)
            Wg = self.M_W[:, idxs]  # [P, Dg]
            moment_V = self.compute_moments_VTV(self.M_V, self.Omega_V, self.S_V)
            S_Zi = torch.linalg.inv((1.0 / tau) * moment_V + torch.eye(self.K, device=self.device))
            M_Zi = (1.0 / tau) * S_Zi @ self.M_V.t() @ Wg
            Omega_Zi = (1.0 / group_moments['mean_gammas'][g]) * torch.linalg.inv(self.Omega_inv_g[g])
            self.M_Z.append(M_Zi)
            self.Omega_Z.append(Omega_Zi)
            self.S_Z.append(S_Zi)
        # Обновление V
        Omega_V_inv = torch.zeros(self.K, self.K, device=self.device)
        for g, idxs in enumerate(self.group_indices):
            Omega_i_inv = self.Omega_inv_g[g]
            M_Zi = self.M_Z[g]
            S_Zi = self.S_Z[g]
            moment_Z = Omega_i_inv * torch.trace(S_Zi) + M_Zi @ Omega_i_inv @ M_Zi.t()
            Omega_V_inv += group_moments['mean_gammas'][g] * moment_Z
        Omega_V_inv += torch.eye(self.K, device=self.device)
        self.Omega_V = torch.linalg.inv(Omega_V_inv)
        self.M_V = self.M_W @ Omega_inv @ Gamma @ Z.t() @ self.Omega_V
        self.S_V = tau * torch.eye(self.P, device=self.device)

    def _update_posterior_wishart(self, group_moments: Dict[str, Any]) -> None:
        """
        Обновляет параметры постериорного распределения Wishart (Lambda, nu, Omega_inv_g) для каждой группы.
        Аргументы:
            group_moments (dict): словарь с моментами и блочными матрицами
        Возвращает: None
        """
        self.Lambda, self.nu_post = [], []
        for g, idxs in enumerate(self.group_indices):
            Dg = len(idxs)
            Wg = self.M_W[:, idxs]  # [P, Dg]
            Zg = self.M_Z[g]  # [K, Dg]
            resid = Wg - self.M_V @ Zg.T  # [P, Dg]
            Omega_W_g = self.Omega_W[idxs, :][:, idxs]
            moment_resid_mean_part = resid.T @ resid
            Omega_i_inv = self.Omega_inv_g[g]
            D_W = torch.trace(self.S_W) * Omega_W_g
            moment_resid_disp_part = D_W + self.Omega_Z[g] * torch.sum(self.S_Z[g] * self.Omega_V) * torch.trace(self.S_V) + self.M_Z[g].T @ self.Omega_V @self.M_Z[g] * torch.trace(self.S_V)+\
            self.Omega_Z[g] * torch.sum(self.S_Z[g] * (self.M_Z[g].T @ self.M_Z[g]))
            moment_resid = moment_resid_mean_part + moment_resid_disp_part
            moment_Z =  self.Omega_Z[g] * torch.sum(self.S_Z[g] @ Omega_i_inv) + self.M_Z[g] @ Omega_i_inv @ self.M_Z[g].T
            Lambda_i = (1.0 / self.tau) * group_moments['mean_gammas'][g] * moment_resid + group_moments['mean_gammas'][g] * moment_Z + torch.eye(Dg, device=self.device)
            self.Lambda.append(Lambda_i)
            self.nu_post.append(float(self.nu_prior) + self.P + self.K)
        self.Omega_inv_g = [(Dg + self.nu_post[g] - 1) * torch.linalg.inv(self.Lambda[g]) for g, idxs in enumerate(self.group_indices)]

    def _update_posterior_gig(self, group_moments: Dict[str, Any]) -> None:
        """
        Обновляет параметры постериорного GIG (omega_post, chi_post, phi_post) для каждой группы.
        Аргументы:
            group_moments (dict): словарь с моментами и блочными матрицами
        Возвращает: None
        """
        for g, idxs in enumerate(self.group_indices):
            Dg = len(idxs)
            # omega_post
            self.omega_post[g] = self.omega_prior + 0.5 * (self.P + self.K) * Dg
            # chi_post
            self.chi_post[g] = self.chi_prior
            # phi_post
            #Вспомогательные переменные
            Omega_i_inv = self.Omega_inv_g[g]
            Wg = self.M_W[:, idxs]  # [P, Dg]
            Zg = self.M_Z[g]  # [K, Dg]
            resid = Wg - self.M_V @ Zg
            Omega_W_g = self.Omega_W[idxs, :][:, idxs]
            Omega_inv_i = self.Omega_inv_g[g]
            #Подсчеты моментов
            moment_resid_mean_part = resid @ Omega_i_inv @ resid.t()
            moment_resid_W_disp_part = self.S_W * torch.sum(Omega_W_g * Omega_inv_i)
            moment_resid_VZ_disp_part = self.S_V * torch.sum(self.Omega_V * self.S_Z[g]) * torch.sum(self.Omega_Z[g]* Omega_i_inv) + self.M_V @ self.S_Z[g] @self.M_V.T * torch.sum(self.Omega_Z[g]* Omega_i_inv)+\
            self.S_V * torch.sum(self.Omega_V * (self.M_Z[g] @ Omega_inv_i @self.M_Z[g].T))
            moment_resid = moment_resid_mean_part + moment_resid_W_disp_part + moment_resid_VZ_disp_part
            tr_resid = torch.trace(moment_resid)
            # Момент: <Z_i Omega_i^{-1} Z_i^T>
            moment_Z = Zg @ Omega_i_inv @ Zg.t() + torch.trace(self.S_Z[g]) * Omega_i_inv
            tr_Z = torch.trace(moment_Z)
            self.phi_post[g] = self.phi_prior + (1.0 / self.tau) * tr_resid + tr_Z
    def m_step(self, post: Dict[str, Any]) -> None:
        """
        M-шаг: обновляет гиперпараметры модели на основе результатов E-шагa.
        
        Аргументы:
            post (Dict[str, Any]): Словарь с результатами E-шагa, включая моменты и параметры постериорного распределения.
        """
        # Теперь только обновление общего прайора
        self.update_gig_prior(post['mean_gammas'], post['mean_inv_gammas'], post['mean_log_gammas'])
        # Обновление tau (дисперсия шума)
        # ... (см. 4.2, зависит от задачи)
    def fit(self, X: Tensor, Y: Tensor, num_iter: int = 10) -> None:
        """
        Обучает модель с использованием EM-алгоритма.
        
        Аргументы:
            X (Tensor): Матрица признаков (N, D).
            Y (Tensor): Матрица откликов (N, P).
            num_iter (int): Количество итераций EM-алгоритма.
        """
        X = X.T
        for _ in range(num_iter):
            post = self.e_step(X, Y)
            self.m_step(post)
            self._set_flat_params(post["M_W"])

    def predict(self, X: Tensor) -> Tensor:
        """
        Выполняет предсказание на основе обученной модели.
        
        Аргументы:
            X (Tensor): Матрица признаков (N, D).
        
        Возвращает:
            Tensor: Предсказания модели (N, P).
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(X).T
