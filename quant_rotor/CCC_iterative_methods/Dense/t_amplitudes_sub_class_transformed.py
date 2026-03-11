from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationParams_Transformed:
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
class TensorData_Transformed:
    t_a_i_tensor: np.ndarray
    t_ab_ij_tensor: np.ndarray
    h_full: np.ndarray
    v_full_xy: np.ndarray
    v_full_yx: np.ndarray
    v_full_per: np.ndarray = field(default=None)


class QuantumSimulation_Transformed:
    def __init__(
        self, params: SimulationParams_Transformed, tensors: TensorData_Transformed
    ):
        self.params = params
        self.tensors = tensors

    def A_term(self, a_upper, a_site):
        return np.hstack((-self.tensors.t_a_i_tensor[a_site], np.identity(a_upper)))

    def B_term(self, b_lower, b_site):
        return np.vstack((np.identity(b_lower), self.tensors.t_a_i_tensor[b_site]))

    def A_term_big(self, a_site):
        i, p = self.params.i, self.params.p

        vec = np.vstack((np.identity(i), -self.tensors.t_a_i_tensor[a_site]))

        return np.hstack((vec, np.identity(p)[:, 1:]))

    def B_term_big(self, b_site):
        i, p = self.params.i, self.params.p

        vec = np.vstack((np.identity(i), self.tensors.t_a_i_tensor[b_site]))

        return np.hstack((vec, np.identity(p)[:, 1:]))

    def h_term(self, h_upper, h_lower):
        a_h_shift = [
            self.params.i if a_check == self.params.a else 0
            for a_check in (h_upper, h_lower)
        ]
        return self.tensors.h_full[
            a_h_shift[0] : h_upper + a_h_shift[0], a_h_shift[1] : h_lower + a_h_shift[1]
        ]

    def h_bar_term(self, x: int, h_upper: int, h_lower: int):

        i, p, a = self.params.i, self.params.p, self.params.a

        K_bar = np.einsum(
            "ps, sq, qr -> pr",
            self.A_term_big(x),
            self.h_term(p, p),
            self.B_term_big(x),
        )
        # K_bar -= np.eye(p) * K_bar[0, 0]

        a_h_shift = [i if a_check == a else 0 for a_check in (h_upper, h_lower)]
        return K_bar[
            a_h_shift[0] : h_upper + a_h_shift[0], a_h_shift[1] : h_lower + a_h_shift[1]
        ]

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

    def v_bar_term(
        self,
        idx_up_1: int,
        idx_up_2: int,
        idx_low_1: int,
        idx_low_2: int,
        x: int,
        y: int,
    ):

        i, p, a = self.params.i, self.params.p, self.params.a

        V_xy_bar = np.einsum(
            "pP, qQ, PQRS, Rr, Ss -> pqrs",
            self.A_term_big(x),
            self.A_term_big(y),
            self.v_term(p, p, p, p, x, y),
            self.B_term_big(x),
            self.B_term_big(y),
        )

        # for indx_1 in range(p):
        #     for indx_2 in range(p):
        #         V_xy_bar[indx_1, indx_2, indx_1, indx_2] -= V_xy_bar[0, 0, 0, 0]
        #         V_yx_bar[indx_1, indx_2, indx_1, indx_2] -= V_yx_bar[0, 0, 0, 0]

        a_v_shift = [
            i if a_check == a else 0
            for a_check in (idx_up_1, idx_up_2, idx_low_1, idx_low_2)
        ]

        return V_xy_bar[
            a_v_shift[0] : idx_up_1 + a_v_shift[0],
            a_v_shift[1] : idx_up_2 + a_v_shift[1],
            a_v_shift[2] : idx_low_1 + a_v_shift[2],
            a_v_shift[3] : idx_low_2 + a_v_shift[3],
        ]

    def t_term_2(self, t_site_1, t_site_2):
        return self.tensors.t_ab_ij_tensor[t_site_1, t_site_2]

    def e_bar_c(self, x: int, y: int):

        a, i = self.params.a, self.params.i

        return np.einsum(
            "ijab, abij",
            self.v_bar_term(
                i,
                i,
                a,
                a,
                x,
                y,
            ),
            self.t_term_2(x, y),
        )

    def E_c(self, x: int):
        site = self.params.site
        term = 0
        for z in range(site):
            if x != z:
                term += self.e_bar_c(x, z)
        return term

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

    def residual_single(self, x: int):
        i, p, a, site = self.params.i, self.params.p, self.params.a, self.params.site

        term = self.h_bar_term(x, a, i)
        for z in range(site):
            if x != z:
                term += np.einsum(
                    "alcd, cdil -> ai",
                    self.v_bar_term(a, i, a, a, x, z),
                    self.t_term_2(x, z),
                )
                for w in range(site):
                    if (w != x) and (w != z):
                        term += np.einsum(
                            "acik, klcl -> ai",
                            self.t_term_2(x, w),
                            self.v_bar_term(i, i, i, a, z, w),
                        )
        return term

    def residual_double_sym(self, x: int, y: int):

        i, p, a, site = self.params.i, self.params.p, self.params.a, self.params.site

        term = self.v_bar_term(a, a, i, i, x, y)

        # print(term)
        term += np.einsum(
            "abcd, cdij -> abij", self.v_bar_term(a, a, a, a, x, y), self.t_term_2(x, y)
        )

        # print("Term_1", np.einsum("abcd, cdij -> abij", v_bar_term(x, y, a, a, a, a), t_2_func(x, y)))
        for z in range(site):
            for w in range(site):
                if z not in {x, y} and w not in {x, y} and z != w:
                    term += np.einsum(
                        "klcd, acik, bdjl -> abij",
                        self.v_bar_term(i, i, a, a, z, w),
                        self.t_term_2(x, z),
                        self.t_term_2(y, w),
                    )

        term -= self.t_term_2(x, y) * (self.E_c(x) + self.E_c(y) - self.e_bar_c(x, y))

        return term

    def residual_double_non_sym_1(self, x: int, y: int):

        i, p, a, site = self.params.i, self.params.p, self.params.a, self.params.site

        term = np.einsum(
            "ac, cbij -> abij", self.h_bar_term(x, a, a), self.t_term_2(x, y)
        )

        # print( h_bar_term(x, a, a))

        for z in range(site):
            if z not in {x, y}:
                term += np.einsum(
                    "acik, kbcj -> abij",
                    self.t_term_2(x, z),
                    self.v_bar_term(i, a, a, i, z, y),
                )
                term += np.einsum(
                    "bldl, adij -> abij",
                    self.v_bar_term(a, i, a, i, y, z),
                    self.t_term_2(x, y),
                )
        return term

    def residual_double_total(self, x_d: int, y_d: int) -> np.ndarray:
        a, i = self.params.a, self.params.i

        return (
            self.residual_double_sym(x_d, y_d)
            + self.residual_double_non_sym_1(x_d, y_d)
            + self.residual_double_non_sym_1(y_d, x_d)
            .reshape(a, a)
            .T.reshape(a, a, i, i)
        )
