from dataclasses import dataclass

import numpy as np
import opt_einsum as oe
import scipy.sparse as sp


@dataclass
class SimulationParams:
    a: int
    i: int
    p: int
    site: int
    state: int
    gap: bool
    gap_site: int
    epsilon: np.ndarray
    periodic: bool = True

@dataclass
class TensorData:
    t_a_i_tensor: np.ndarray
    t_ab_ij_tensor: np.ndarray
    h_full: np.ndarray
    v_full_xy: np.ndarray
    v_full_yx: np.ndarray

@dataclass
class PrecalcalculatedTerms:
    a_term: np.ndarray = None
    b_term: np.ndarray = None
    aa_term: np.ndarray = None
    bb_term: np.ndarray = None
    h_pp: np.ndarray = None
    h_pa: np.ndarray = None
    h_ip: np.ndarray = None

class QuantumSimulation:
    def __init__(self, params: SimulationParams, tensors: TensorData, terms: PrecalcalculatedTerms):
        self.params = params
        self.tensors = tensors
        self.terms = terms

    def A_term(self, a_upper):
        return np.hstack(
            (
                -self.tensors.t_a_i_tensor[0].reshape(self.params.a, self.params.i),
                np.identity(a_upper),
            )
        )

    def B_term(self, b_lower):
        return np.concatenate(
            (
                np.ones(b_lower),
                self.tensors.t_a_i_tensor[0],
            )
        )

    def h_term(self, h_upper, h_lower):
        a_h_shift = [self.params.i if a_check == self.params.a else 0 for a_check in (h_upper, h_lower)]
        return self.tensors.h_full[a_h_shift[0]:h_upper + a_h_shift[0], a_h_shift[1]:h_lower + a_h_shift[1]]

    def v_term(self, v_upper_1, v_upper_2, v_lower_1, v_lower_2, v_site_1, v_site_2):
        if self.params.periodic:
            if (v_site_2 - v_site_1) == 1 or (v_site_2 - v_site_1) == (
                self.params.site - 1
            ):
                a_v_shift = [
                    self.params.i if a_check == self.params.a else 0
                    for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)
                ]
                return self.tensors.v_full_xy[
                    a_v_shift[0] : v_upper_1 + a_v_shift[0],
                    a_v_shift[1] : v_upper_2 + a_v_shift[1],
                    a_v_shift[2] : v_lower_1 + a_v_shift[2],
                    a_v_shift[3] : v_lower_2 + a_v_shift[3],
                ]
            elif v_site_1 - v_site_2 == 1 or (v_site_1 - v_site_2) == (
                self.params.site - 1
            ):
                a_v_shift = [
                    self.params.i if a_check == self.params.a else 0
                    for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)
                ]
                return self.tensors.v_full_yx[
                    a_v_shift[0] : v_upper_1 + a_v_shift[0],
                    a_v_shift[1] : v_upper_2 + a_v_shift[1],
                    a_v_shift[2] : v_lower_1 + a_v_shift[2],
                    a_v_shift[3] : v_lower_2 + a_v_shift[3],
                ]
            else:
                return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))
        else:
            if self.params.gap and (
                (
                    v_site_1 == self.params.gap_site
                    and v_site_2 == self.params.gap_site + 1
                )
                or (
                    v_site_1 == self.params.gap_site + 1
                    and v_site_2 == self.params.gap_site
                )
            ):
                return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))

            if v_site_2 - v_site_1 == 1:
                a_v_shift = [
                    self.params.i if a_check == self.params.a else 0
                    for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)
                ]
                return self.tensors.v_full_xy[
                    a_v_shift[0] : v_upper_1 + a_v_shift[0],
                    a_v_shift[1] : v_upper_2 + a_v_shift[1],
                    a_v_shift[2] : v_lower_1 + a_v_shift[2],
                    a_v_shift[3] : v_lower_2 + a_v_shift[3],
                ]
            elif v_site_1 - v_site_2 == 1:
                a_v_shift = [self.params.i if a_check == self.params.a else 0 for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)]
                return self.tensors.v_full_yx[
                    a_v_shift[0] : v_upper_1 + a_v_shift[0],
                    a_v_shift[1] : v_upper_2 + a_v_shift[1],
                    a_v_shift[2] : v_lower_1 + a_v_shift[2],
                    a_v_shift[3] : v_lower_2 + a_v_shift[3],
                ]
            else:
                return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))

    def t_term(self, t_site_1, t_site_2):
        return self.tensors.t_ab_ij_tensor[t_site_1, t_site_2]

    def update_one(self, r_1_value):
        a, i, eps = self.params.a, self.params.i, self.params.epsilon
        update = np.zeros((a), dtype=complex)
        for u_a in range(a):
            update[u_a] = 1 / (eps[u_a + i] - eps[0])
        return update * r_1_value

    def update_two(self, r_2_value):
        a, i, eps = self.params.a, self.params.i, self.params.epsilon
        update = np.zeros((a, a), dtype=complex)
        for u_a in range(a):
            for u_b in range(a):
                update[u_a, u_b] = 1 / (eps[u_a + i] + eps[u_b + i] - eps[0] - eps[0])
        return update * r_2_value

    def residual_single(self) -> np.ndarray:
        """
        Computes the single excitation residual R^{a}_{i} for site x_s = 0
        using optimized einsum contractions.

        Returns
        -------
        R_single : np.ndarray
            Residual tensor of shape (a, i) for the single excitation.
        """
        # Fixed site index
        x = 0

        # Unpack parameters
        site, a, p, i = self.params.site, self.params.a, self.params.p, self.params.i

        A_ap, B_qi, BB_Qi = self.terms.a_term, self.terms.b_term, self.terms.bb_term
        R_single = np.zeros((a), dtype=complex)

        H_pq = self.terms.h_pp

        # Term 1: A H B

        R_single += A_ap @ (H_pq @ B_qi)

        # Terms from other sites
        for z in range(site):
            if z != x:
                T_Ci = self.t_term(x, z).reshape(a**2)
                V_pQ = self.v_term(p, i, p, p, x, z).reshape(p, p**2)
                V_pC = self.v_term(p, i, a, a, x, z).reshape(p, a**2)

                # Term 2: A V T
                R_single += A_ap @ (V_pC @ T_Ci)

                # Term 3: A V B B
                R_single += A_ap @ (V_pQ @ BB_Qi)

                for w in range(site):
                    if w not in {x, z}:
                        T_ac = self.t_term(x, w).reshape(a, a)
                        V_cq = self.v_term(i, i, a, p, w, z).reshape(a, p)
                        R_single += T_ac @ (V_cq @ B_qi)

        return R_single

    def residual_double_sym(self, x: int, y: int) -> np.ndarray:
        """
        Computes symmetric double residual R^{ab}_{ij}(0, y)
        assuming site x = 0 is fixed and y varies.
        Uses optimized tensor contractions via opt_einsum.
        """

        site, a, p, i = self.params.site, self.params.a, self.params.p, self.params.i
        B, AA_BQ, BB_R = self.terms.b_term, self.terms.aa_term, self.terms.bb_term

        V_QR = self.v_term(p, p, p, p, x, y).reshape(p**2, p**2)
        # Term 1: A ⊗ A · V · B ⊗ B

        R = (AA_BQ @ (V_QR @ BB_R)).reshape(a, a)

        T_C_flat = self.t_term(x, y).reshape(a**2)
        T_C = self.t_term(x, y)
        V_QC = self.v_term(p, p, a, a, x, y).reshape(p**2, a**2)
        V_pq = self.v_term(i, i, p, p, x, y).reshape(p, p)

        R += (AA_BQ @ (V_QC @ T_C_flat)).reshape(a, a)
        R -= T_C * ((V_pq @ B) @ B)

        V_cd = self.v_term(i, i, a, a, x, y).reshape(a, a)
        scalar = np.sum(V_cd * T_C)
        R -= T_C * scalar

        # Term 5: all connected permutations
        for w in range(site):
            for z in range(site):
                if z not in {x, y} and w not in {x, y} and z != w:
                    V_cd = self.v_term(i, i, a, a, z, w).reshape(a, a)
                    T_xz_1 = self.t_term(x, z)
                    T_yw_1 = self.t_term(y, w)

                    R += T_xz_1 @ V_cd @ T_yw_1.T
        return R

    def residual_double_non_sym_1(self, x: int, y: int) -> np.ndarray:
        """
        Computes asymmetric residual R^{ab}_{ij} for fixed x = 0 and variable y.
        Corresponds to the first non-symmetric contraction path using optimized einsums.
        """
        site, a, p, i = self.params.site, self.params.a, self.params.p, self.params.i
        A, B = self.terms.a_term, self.terms.b_term
        h_pc, h_p = self.terms.h_pa, self.terms.h_ip

        T_cb = self.t_term(x, y)

        R = A @ h_pc @ T_cb
        R -= T_cb * (h_p @ B)

        for z in range(site):
            if z != x and z != y:
                V_ipap = self.v_term(i, p, a, p, z, y).reshape(p, a, p)
                T_xz = self.t_term(x, z)
                T_xy = self.t_term(x, y)
                V_pp = self.v_term(i, i, p, p, x, z).reshape(p, p)
                V_pap = self.v_term(p, i, a, p, y, z).reshape(p, a, p)
                V_cd = self.v_term(i, i, a, a, x, y).reshape(a, a)

                R += T_xz @ ((V_ipap @ B).T @ A.T)
                R += T_cb @ (A @ (V_pap @ B).reshape(p, a)).T
                R -= T_cb * ((V_pp @ B) @ B)

                scalar = np.sum(V_cd * T_xz)
                R -= T_xy * scalar
        return R

    def residual_double_total(self, x: int, y: int) -> np.ndarray:
        return (
            self.residual_double_sym(x, y)
            + self.residual_double_non_sym_1(x, y)
            + self.residual_double_non_sym_1(y, x).T
        )
