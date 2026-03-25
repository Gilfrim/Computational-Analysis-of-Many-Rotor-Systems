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
    h_ia: np.ndarray = None
    V_pppp: np.ndarray = None
    V_ppaa: np.ndarray = None
    V_iipp: np.ndarray = None
    V_iiaa: np.ndarray = None
    V_piaa: np.ndarray = None
    V_pipp: np.ndarray = None
    V_ipap: np.ndarray = None
    V_piap: np.ndarray = None
    V_iiap: np.ndarray = None


class QuantumSimulation:
    def __init__(
        self,
        params: SimulationParams,
        tensors: TensorData,
        terms: PrecalcalculatedTerms,
    ):
        self.params = params
        self.tensors = tensors
        self.terms = terms

    def A_term(self):
        return np.hstack(
            (
                -self.tensors.t_a_i_tensor.reshape(self.params.a, self.params.i),
                np.identity(self.params.a),
            )
        )

    def B_term(self, x):
        return np.concatenate(
            (
                np.ones(self.params.i),
                self.tensors.t_a_i_tensor[x],
            )
        )

    def h_term(self, h_upper, h_lower):
        a_h_shift = [
            self.params.i if a_check == self.params.a else 0
            for a_check in (h_upper, h_lower)
        ]
        return self.tensors.h_full[
            a_h_shift[0] : h_upper + a_h_shift[0], a_h_shift[1] : h_lower + a_h_shift[1]
        ]

    def v_term(self, v_upper_1, v_upper_2, v_lower_1, v_lower_2, v_site_1, v_site_2):

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

    def t_term(self, x, y):
        return self.tensors.t_ab_ij_tensor[np.abs(x - y) - 1]

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

    def residual_single(self, x) -> np.ndarray:
        """
        Computes the single excitation residual R^{a}_{i} for site x_s = 0
        using optimized einsum contractions.

        Returns
        -------
        R_single : np.ndarray
            Residual tensor of shape (a, i) for the single excitation.
        """

        # Unpack parameters
        site, a, periodic = (
            self.params.site,
            self.params.a,
            self.params.periodic,
        )

        A_ap, B_qi, BB_Qi, H_pq, H_ia, V_cq, V_pQ, V_pC = (
            self.terms.a_term,
            self.terms.b_term,
            self.terms.bb_term,
            self.terms.h_pp,
            self.terms.h_ia,
            self.terms.V_iiap,
            self.terms.V_pipp,
            self.terms.V_piaa,
        )

        T_Ci = self.t_term(x, 1).reshape(a**2)
        T_ac_0 = self.t_term(x, 1).reshape(a, a)

        x = 0

        # Term 1: A H B

        R_single += A_ap @ (H_pq @ B_qi)

        # Term 2: A V T
        R_single += A_ap @ (V_pC @ T_Ci)
        R_single += self.t_term(x, 1) @ H_ia
        # R_single += np.einsum("b, ab -> a", H_ia, self.t_term(z))

        # Term 3: A V B B
        R_single += A_ap @ (V_pQ @ BB_Qi)

        R_single += T_ac_0 @ (V_cq @ B_qi)
        # Terms from other sites
        for z in range(2, site):
            T_ac_1 = self.t_term(x, z).reshape(a, a)
            R_single += (T_ac_1 @ (V_cq @ B_qi)) * 2

        return R_single

    def residual_double_sym(self, y: int) -> np.ndarray:
        """
        Computes symmetric double residual R^{ab}_{ij}(0, y)
        assuming site x = 0 is fixed and y varies.
        Uses optimized tensor contractions via opt_einsum.
        """

        site, a, p = self.params.site, self.params.a, self.params.p
        B_pi, AA_BQ, BB_R, V_QR, V_QC, V_pq, V_cd = (
            self.terms.b_term,
            self.terms.aa_term,
            self.terms.bb_term,
            self.terms.V_pppp,
            self.terms.V_ppaa,
            self.terms.V_iipp,
            self.terms.V_iiaa,
        )

        # Term 1: A ⊗ A · V · B ⊗ B

        x = 0

        R = (AA_BQ @ (V_QR @ BB_R)).reshape(a, a)

        T_C_flat = self.t_term(x, y).reshape(a**2)
        T_C = self.t_term(x, y)

        R += (AA_BQ @ (V_QC @ T_C_flat)).reshape(a, a)
        R -= T_C * ((V_pq @ B_pi) @ B_pi)

        R -= T_C * np.sum(V_cd * T_C)

        # Term 5: all connected permutations
        for w in range(site):
            for z in range(site):
                if z not in {x, y} and w not in {x, y} and z != w:
                    T_xz_1 = self.t_term(x, z)
                    T_yw_1 = self.t_term(y, w)

                    R += T_xz_1 @ V_cd @ T_yw_1.T
        return R

    def residual_double_non_sym_1(self, x: int, y: int) -> np.ndarray:
        """
        Computes asymmetric residual R^{ab}_{ij} for fixed x = 0 and variable y.
        Corresponds to the first non-symmetric contraction path using optimized einsums.
        """
        site, a, p = self.params.site, self.params.a, self.params.p
        A_ap, B_pi, h_pc, h_p, V_pp, V_pap, V_cd, V_ipap = (
            self.terms.a_term,
            self.terms.b_term,
            self.terms.h_pa,
            self.terms.h_ip,
            self.terms.V_iipp,
            self.terms.V_piap,
            self.terms.V_iiaa,
            self.terms.V_ipap,
        )

        T_cb = self.t_term(x, y)

        R = A_ap @ h_pc @ T_cb
        R -= T_cb * (h_p @ B_pi)

        for z in range(site):
            if z != x and z != y:
                T_xz = self.t_term(x, z)
                T_xy = self.t_term(x, y)

                R += T_xz @ ((V_ipap @ B_pi).T @ A_ap.T)
                R += T_cb @ (A_ap @ (V_pap @ B_pi).reshape(p, a)).T
                R -= T_cb * ((V_pp @ B_pi) @ B_pi)

                R -= T_xy * np.sum(V_cd * T_xz)

        return R

    def residual_double_total(self, x: int, y: int) -> np.ndarray:

        return (
            self.residual_double_sym(x, y)
            + self.residual_double_non_sym_1(x, y)
            + self.residual_double_non_sym_1(y, x).T
        )
