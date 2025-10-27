from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationParams:
    a: int
    i: int
    p: int
    site: int
    state: int
    i_method: int
    gap: bool
    gap_site: int
    epsilon: np.ndarray
    periodic: bool=False
    double: bool = False

@dataclass
class TensorData:
    t_a_i_tensor: np.ndarray
    t_ab_ij_tensor: np.ndarray
    h_full: np.ndarray
    v_full: np.ndarray
    v_full_per: np.ndarray = field(default=None)

class QuantumSimulation:
    def __init__(self, params: SimulationParams, tensors: TensorData):
        self.params = params
        self.tensors = tensors

    def A_term(self, a_upper, a_site):
        return np.hstack((-self.tensors.t_a_i_tensor[a_site], np.identity(a_upper)))

    def B_term(self, b_lower, b_site):
        return np.vstack((np.identity(b_lower), self.tensors.t_a_i_tensor[b_site]))

    def h_term(self, h_upper, h_lower):
        a_h_shift = [self.params.i if a_check == self.params.a else 0 for a_check in (h_upper, h_lower)]
        return self.tensors.h_full[a_h_shift[0]:h_upper + a_h_shift[0], a_h_shift[1]:h_lower + a_h_shift[1]]

    def v_term(self, v_upper_1, v_upper_2, v_lower_1, v_lower_2, v_site_1, v_site_2):
        if self.params.periodic and self.params.double:
            if abs(v_site_1 - v_site_2) == 1:
                a_v_shift = [
                    self.params.i if a_check == self.params.a else 0
                    for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)
                ]
                return self.tensors.v_full[
                    a_v_shift[0] : v_upper_1 + a_v_shift[0],
                    a_v_shift[1] : v_upper_2 + a_v_shift[1],
                    a_v_shift[2] : v_lower_1 + a_v_shift[2],
                    a_v_shift[3] : v_lower_2 + a_v_shift[3],
                ]
            elif abs(v_site_1 - v_site_2) == (self.params.site - 1):
                a_v_shift = [
                    self.params.i if a_check == self.params.a else 0
                    for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)
                ]
                return self.tensors.v_full_per[
                    a_v_shift[0] : v_upper_1 + a_v_shift[0],
                    a_v_shift[1] : v_upper_2 + a_v_shift[1],
                    a_v_shift[2] : v_lower_1 + a_v_shift[2],
                    a_v_shift[3] : v_lower_2 + a_v_shift[3],
                ]

            else:
                return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))
        if self.params.periodic:
            if abs(v_site_1 - v_site_2) == 1 or abs(v_site_1 - v_site_2) == (self.params.site - 1):
                a_v_shift = [self.params.i if a_check == self.params.a else 0 for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)]
                return self.tensors.v_full[
                    a_v_shift[0]:v_upper_1 + a_v_shift[0],
                    a_v_shift[1]:v_upper_2 + a_v_shift[1],
                    a_v_shift[2]:v_lower_1 + a_v_shift[2],
                    a_v_shift[3]:v_lower_2 + a_v_shift[3]
                ]
            else:
                return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))
        else:
            if self.params.gap and ((v_site_1 == self.params.gap_site and v_site_2 == self.params.gap_site + 1) or (v_site_1 == self.params.gap_site + 1 and v_site_2 == self.params.gap_site)):
                return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))

            if abs(v_site_1 - v_site_2) == 1:
                a_v_shift = [self.params.i if a_check == self.params.a else 0 for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)]
                return self.tensors.v_full[
                    a_v_shift[0]:v_upper_1 + a_v_shift[0],
                    a_v_shift[1]:v_upper_2 + a_v_shift[1],
                    a_v_shift[2]:v_lower_1 + a_v_shift[2],
                    a_v_shift[3]:v_lower_2 + a_v_shift[3]
                ]
            else:
                return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))

    def t_term(self, t_site_1, t_site_2):
        return self.tensors.t_ab_ij_tensor[t_site_1, t_site_2]

    def update_one(self, r_1_value):
        a, i, eps = self.params.a, self.params.i, self.params.epsilon
        update = np.zeros((a, i), dtype=complex)
        for u_a in range(a):
            for u_i in range(i):
                update[u_a, u_i] = 1 / (eps[u_a + i] - eps[u_i])
        return update * r_1_value

    def update_two(self, r_2_value):
        a, i, eps = self.params.a, self.params.i, self.params.epsilon
        update = np.zeros((a, a, i, i), dtype=complex)
        for u_a in range(a):
            for u_b in range(a):
                for u_i in range(i):
                    for u_j in range(i):
                        update[u_a, u_b, u_i, u_j] = 1 / (eps[u_a + i] + eps[u_b + i] - eps[u_i] - eps[u_j])
        return update * r_2_value

    def residual_single(self, x_s: int) -> np.ndarray:
        """Calculates R^{a}_{i}(x) singles equation"""
        site, i_method, a, i, p = self.params.site, self.params.i_method, self.params.a, self.params.i, self.params.p

        R_single = np.zeros((a, i), dtype = complex)

        R_single += np.einsum("ap, pq, qi->ai", self.A_term(a, x_s), self.h_term(p, p), self.B_term(i, x_s))

        for z_s in range(site):
            if z_s != x_s:
                if i_method >= 1:
                    R_single += np.einsum("ap, plcd, cdil->ai", self.A_term(a, x_s), self.v_term(p, i, a, a, x_s, z_s), self.t_term(x_s, z_s))

                R_single += np.einsum("ap, plqs, qi, sl->ai", self.A_term(a, x_s), self.v_term(p, i, p, p, x_s, z_s), self.B_term(i, x_s), self.B_term(i, z_s))

        return R_single

    def residual_double_sym(self, x_d: int, y_d: int) -> np.ndarray:
        site, i_method, p, i, a = self.params.site, self.params.i_method, self.params.p, self.params.i, self.params.a
        R = np.zeros((a, a, i, i), dtype = complex)

        if i_method >= 1:
            R += np.einsum("ap, bq, pqrs, ri, sj->abij", self.A_term(a, x_d), self.A_term(a, y_d), self.v_term(p, p, p, p, x_d, y_d), self.B_term(i, x_d), self.B_term(i, y_d))

            if i_method >= 2:
                R += np.einsum("ap, bq, pqcd, cdij->abij", self.A_term(a, x_d), self.A_term(a, x_d), self.v_term(p, p, a, a, x_d, y_d), self.t_term(x_d, y_d))

                R -= np.einsum("abkl, klpq, pi, qj->abij", self.t_term(x_d, y_d), self.v_term(i, i, p, p, x_d, y_d), self.B_term(i, x_d), self.B_term(i, y_d))

                if i_method == 3:
                    R -= np.einsum("abkl, klcd, cdij->abij", self.t_term(x_d, y_d), self.v_term(i, i, a, a, x_d, y_d), self.t_term(x_d, y_d))

                    if site >= 4:
                        for z in range(site):
                            for w in range(site):
                                if z not in {x_d, y_d} and w not in {x_d, y_d} and z != w:
                                    R += np.einsum("klcd, acik, bdjl->abij", self.v_term(i, i, a, a, z, w), self.t_term(x_d, z), self.t_term(y_d, w))
        return R

    def residual_double_non_sym_1(self, x_d: int, y_d: int) -> np.ndarray:
        site, i_method, p, i, a = self.params.site, self.params.i_method, self.params.p, self.params.i, self.params.a
        R = np.zeros((a, a, i, i), dtype = complex)

        if i_method >= 1:
            R += np.einsum("ap, pc, cbij->abij", self.A_term(a, x_d), self.h_term(p, a), self.t_term(x_d, y_d))

            R -= np.einsum("abkj, kp, pi->abij", self.t_term(x_d, y_d), self.h_term(i, p), self.B_term(i, x_d))

            if i_method >= 2:
                for z in range(site):
                    if z != x_d and z != y_d:
                        R += np.einsum("acik, krcs, br, sj->abij", self.t_term(x_d, z), self.v_term(i, p, a, p, z, y_d), self.A_term(a, y_d), self.B_term(i, y_d))

                        R += np.einsum("bq, qlds, adij, sl->abij", self.A_term(a, y_d), self.v_term(p, i, a, p, y_d, z), self.t_term(x_d, y_d), self.B_term(i, z))

                        R -= np.einsum("abkj, lkrp, pi, rl->abij", self.t_term(x_d, y_d), self.v_term(i, i, p, p, z, x_d), self.B_term(i, x_d), self.B_term(i, z))
        return R

    def residual_double_non_sym_2(self, x_d: int, y_d: int) -> np.ndarray:
        site, i_method, p, i, a = self.params.site, self.params.i_method, self.params.p, self.params.i, self.params.a
        R = np.zeros((a, a, i, i), dtype = complex)

        if i_method >= 1:
            R += np.einsum("bp, pc, caji->baji", self.A_term(a, y_d), self.h_term(p, a), self.t_term(y_d, x_d))

            R -= np.einsum("baki, kp, pj->baji", self.t_term(y_d, x_d), self.h_term(i, p), self.B_term(i, y_d))

            if i_method >= 2:
                for z in range(site):
                    if z != x_d and z != y_d:
                        R += np.einsum("bcjk, krcs, ar, si->baji", self.t_term(y_d, z), self.v_term(i, p, a, p, z, x_d), self.A_term(a, x_d), self.B_term(i, x_d))

                        R += np.einsum("aq, qlds, bdji, sl->baji", self.A_term(a, x_d), self.v_term(p, i, a, p, x_d, z), self.t_term(y_d, x_d), self.B_term(i, z))

                        R -= np.einsum("baki, lkrp, pj, rl->baji", self.t_term(y_d, x_d), self.v_term(i, i, p, p, z, y_d), self.B_term(i, y_d), self.B_term(i, z))
        return R

    def residual_double_total(self, x_d: int, y_d: int) -> np.ndarray:
        return (self.residual_double_sym(x_d, y_d) + self.residual_double_non_sym_1(x_d, y_d) + self.residual_double_non_sym_2(x_d, y_d))

    def transformation_test(self):
        state = self.params.state
        h_full = self.tensors.h_full.copy()
        v_full = self.tensors.v_full.copy()

        for p in range(state):
            for q in range(state):
                if p != q:
                    h_full[p, q] = 0.1 / abs(p - q)

        _, U = np.linalg.eigh(h_full)
        h_trans = U.T @ h_full @ U
        v_trans = np.einsum("ip, jr, prqs, qk, sl->ijkl", U.T, U.T, v_full, U, U)

        return h_trans, v_trans
