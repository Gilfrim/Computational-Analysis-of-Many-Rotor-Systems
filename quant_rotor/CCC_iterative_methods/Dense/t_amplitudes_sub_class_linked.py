from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationParams_linked:
    a: int
    i: int
    p: int
    site: int
    state: int
    i_method: int
    gap: bool
    gap_site: int
    epsilon: np.ndarray
    periodic: bool = False
    double: bool = False


@dataclass
class TensorData_linked:
    t_a_i_tensor: np.ndarray
    t_ab_ij_tensor: np.ndarray
    h_full: np.ndarray
    v_full_xy: np.ndarray
    v_full_yx: np.ndarray


class QuantumSimulation_linked:
    def __init__(self, params: SimulationParams_linked, tensors: TensorData_linked):
        self.params = params
        self.tensors = tensors

    def A_term(self, a_upper, a_site):
        return np.hstack((-self.tensors.t_a_i_tensor[a_site], np.identity(a_upper)))

    def B_term(self, b_lower, b_site):
        return np.vstack((np.identity(b_lower), self.tensors.t_a_i_tensor[b_site]))

    def h_term(self, h_upper, h_lower):
        a_h_shift = [
            self.params.i if a_check == self.params.a else 0
            for a_check in (h_upper, h_lower)
        ]
        return self.tensors.h_full[
            a_h_shift[0] : h_upper + a_h_shift[0], a_h_shift[1] : h_lower + a_h_shift[1]
        ]

    def v_term(
        self,
        idx_up_1,
        idx_up_2,
        idx_low_1,
        idx_low_2,
        x,
        y,
    ):

        a_v_shift = [
            self.params.i if a_check == self.params.a else 0
            for a_check in (idx_up_1, idx_up_2, idx_low_1, idx_low_2)
        ]
        if (x - y) == 1:

            return self.tensors.v_full_yx[
                a_v_shift[0] : idx_up_1 + a_v_shift[0],
                a_v_shift[1] : idx_up_2 + a_v_shift[1],
                a_v_shift[2] : idx_low_1 + a_v_shift[2],
                a_v_shift[3] : idx_low_2 + a_v_shift[3],
            ]

        if (y - x) == 1:

            return self.tensors.v_full_xy[
                a_v_shift[0] : idx_up_1 + a_v_shift[0],
                a_v_shift[1] : idx_up_2 + a_v_shift[1],
                a_v_shift[2] : idx_low_1 + a_v_shift[2],
                a_v_shift[3] : idx_low_2 + a_v_shift[3],
            ]

        else:
            return np.zeros((idx_up_1, idx_up_2, idx_low_1, idx_low_2))

    # def v_term(
    #     self,
    #     v_upper_1,
    #     v_upper_2,
    #     v_lower_1,
    #     v_lower_2,
    #     v_site_1,
    #     v_site_2,
    # ):
    #     if self.params.periodic:
    #         if (v_site_2 - v_site_1) == 1 or (v_site_2 - v_site_1) == (
    #             self.params.site - 1
    #         ):
    #             a_v_shift = [
    #                 self.params.i if a_check == self.params.a else 0
    #                 for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)
    #             ]
    #             return self.tensors.v_full_xy[
    #                 a_v_shift[0] : v_upper_1 + a_v_shift[0],
    #                 a_v_shift[1] : v_upper_2 + a_v_shift[1],
    #                 a_v_shift[2] : v_lower_1 + a_v_shift[2],
    #                 a_v_shift[3] : v_lower_2 + a_v_shift[3],
    #             ]
    #         elif v_site_1 - v_site_2 == 1 or (v_site_1 - v_site_2) == (
    #             self.params.site - 1
    #         ):
    #             a_v_shift = [
    #                 self.params.i if a_check == self.params.a else 0
    #                 for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)
    #             ]
    #             return self.tensors.v_full_yx[
    #                 a_v_shift[0] : v_upper_1 + a_v_shift[0],
    #                 a_v_shift[1] : v_upper_2 + a_v_shift[1],
    #                 a_v_shift[2] : v_lower_1 + a_v_shift[2],
    #                 a_v_shift[3] : v_lower_2 + a_v_shift[3],
    #             ]
    #         else:
    #             return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))
    #     else:
    #         # if self.params.gap and (
    #         #     (
    #         #         v_site_1 == self.params.gap_site
    #         #         and v_site_2 == self.params.gap_site + 1
    #         #     )
    #         #     or (
    #         #         v_site_1 == self.params.gap_site + 1
    #         #         and v_site_2 == self.params.gap_site
    #         #     )
    #         # ):
    #         #     return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))

    #         if v_site_2 - v_site_1 == 1:
    #             a_v_shift = [
    #                 self.params.i if a_check == self.params.a else 0
    #                 for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)
    #             ]
    #             return self.tensors.v_full_xy[
    #                 a_v_shift[0] : v_upper_1 + a_v_shift[0],
    #                 a_v_shift[1] : v_upper_2 + a_v_shift[1],
    #                 a_v_shift[2] : v_lower_1 + a_v_shift[2],
    #                 a_v_shift[3] : v_lower_2 + a_v_shift[3],
    #             ]
    #         elif v_site_1 - v_site_2 == 1:
    #             a_v_shift = [
    #                 self.params.i if a_check == self.params.a else 0
    #                 for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)
    #             ]
    #             return self.tensors.v_full_yx[
    #                 a_v_shift[0] : v_upper_1 + a_v_shift[0],
    #                 a_v_shift[1] : v_upper_2 + a_v_shift[1],
    #                 a_v_shift[2] : v_lower_1 + a_v_shift[2],
    #                 a_v_shift[3] : v_lower_2 + a_v_shift[3],
    #             ]
    #         else:
    #             return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))

    def t_term_1(self, t_site_1):
        return self.tensors.t_a_i_tensor[t_site_1]

    def t_term_2(self, t_site_1, t_site_2):
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
                        update[u_a, u_b, u_i, u_j] = 1 / (
                            eps[u_a + i] + eps[u_b + i] - eps[u_i] - eps[u_j]
                        )
        return update * r_2_value

    def ec1(self, x: int):

        i, a = self.params.i, self.params.a

        return np.einsum("ia, ai", self.h_term(i, a), self.t_term_1(x))

    def ec2(self, x: int, y: int):

        i, a = self.params.i, self.params.a

        term = self.t_term_2(x, y) + np.einsum(
            "ai, bj-> abij", self.t_term_1(x), self.t_term_1(y)
        )

        return np.einsum("ijab, abij", self.v_term(i, i, a, a, x, y), term)

    def et1(self, x: int):

        term = self.ec1(x)

        for z in range(self.params.site):
            if x != z:
                term += self.ec2(x, z)

        return term

    def c1(self, x: int):

        p, a, i = self.params.p, self.params.a, self.params.i

        return np.einsum("aq, qi -> ai", self.h_term(a, p), self.B_term(i, x))

    def c2(self, x: int, z: int):

        p, a, i = self.params.p, self.params.a, self.params.i

        term = np.einsum(
            "alcd, cdil -> ai",
            self.v_term(a, i, a, a, x, z),
            self.t_term_2(x, z),
        )

        term += np.einsum(
            "alqs, qi, sl -> ai",
            self.v_term(a, i, p, p, x, z),
            self.B_term(i, x),
            self.B_term(i, z),
        )
        return term

    def c3(self, x: int, z: int, w: int):

        p, a, i = self.params.p, self.params.a, self.params.i

        return np.einsum(
            "acik, klcp, pl -> ai",
            self.t_term_2(x, w),
            self.v_term(i, i, a, p, w, z),
            self.B_term(i, z),
        )

    def residual_single(self, x: int) -> np.ndarray:
        """Calculates R^{a}_{i}(x) singles equation"""
        site, a, i = (
            self.params.site,
            self.params.a,
            self.params.i,
        )

        R_single = self.c1(x)
        for z in range(site):
            if z != x:
                R_single += self.c2(x, z)
                for w in range(site):
                    if (w != x) and (w != z):
                        R_single += self.c3(x, z, w)
        R_single -= self.et1(x) * self.t_term_1(x)

        return R_single

    def residual_double_sym(self, x: int, y: int) -> np.ndarray:
        site, p, i, a = (
            self.params.site,
            self.params.p,
            self.params.i,
            self.params.a,
        )

        R_double = np.einsum(
            "abrs, ri, sj -> abij",
            self.v_term(a, a, p, p, x, y),
            self.B_term(i, x),
            self.B_term(i, y),
        )
        R_double += np.einsum(
            "abcd, cdij -> abij", self.v_term(a, a, a, a, x, y), self.t_term_2(x, y)
        )
        for z in range(site):
            for w in range(site):
                if z not in {x, y} and w not in {x, y} and z != w:
                    R_double += np.einsum(
                        "klcd, acik, bdjl -> abij",
                        self.v_term(i, i, a, a, z, w),
                        self.t_term_2(y, w),
                    )

        R_double -= self.t_term_2(x, y) * (self.et1(x) + self.et1(y) - self.ec2(x, y))
        R_double += np.einsum(
            "ai, bj -> abij", self.t_term_1(x), self.t_term_1(y)
        ) * self.ec2(x, y)

        return R_double

    def residual_double_non_sym_1(self, x: int, y: int) -> np.ndarray:
        site, p, i, a = (
            self.params.site,
            self.params.p,
            self.params.i,
            self.params.a,
        )

        term_intermediet = np.zeros((a, i), dtype=complex)

        R_double = np.einsum("ac, cbij -> abij", self.h_term(a, a), self.t_term_2(x, y))

        for z in range(site):
            if (z != x) and (z != y):
                R_double += np.einsum(
                    "acik, kbcs, sj -> abij",
                    self.t_term_2(x, z),
                    self.v_term(i, a, a, p, z, y),
                    self.B_term(i, y),
                )
                R_double += np.einsum(
                    "blds, adij, sl -> abij",
                    self.v_term(a, a, i, i, y, z),
                    self.t_term_2(x, y),
                    self.B_term(i, z),
                )
                term_intermediet += self.c3(x, y, z) + self.c3(x, z, y)

        R_double -= np.einsum(
            "ai, bj -> abij", (self.c2(x, y) + term_intermediet), self.t_term_1(y)
        )

        return R_double

    def residual_double_total(self, x: int, y: int) -> np.ndarray:
        a, i = self.params.a, self.params.i

        return (
            self.residual_double_sym(x, y)
            + self.residual_double_non_sym_1(x, y)
            + self.residual_double_non_sym_1(y, x).reshape(a, a).T.reshape(a, a, i, i)
        )
