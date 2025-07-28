from dataclasses import dataclass
import numpy as np
import opt_einsum as oe

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

@dataclass
class TensorData:
    t_a_i_tensor: np.ndarray
    t_ab_ij_tensor: np.ndarray
    h_full: np.ndarray
    v_full: np.ndarray

@dataclass
class PrecalcalculatedTerms:
    a_term: np.ndarray = None
    b_term: np.ndarray = None
    h_pp: np.ndarray = None
    h_pa: np.ndarray = None
    h_ip: np.ndarray = None
    V_pppp: np.ndarray = None
    V_ppaa: np.ndarray = None
    V_iipp: np.ndarray = None
    V_iiaa: np.ndarray = None
    V_piaa: np.ndarray = None
    V_pipp: np.ndarray = None
    V_ipap: np.ndarray = None
    V_piap: np.ndarray = None

class QuantumSimulation:
    def __init__(self, params: SimulationParams, tensors: TensorData, terms: PrecalcalculatedTerms):
        self.params = params
        self.tensors = tensors
        self.terms = terms

    def A_term(self, a_upper):
        return np.hstack((-self.tensors.t_a_i_tensor, np.identity(a_upper)))

    def B_term(self, b_lower):
        return np.vstack((np.identity(b_lower), self.tensors.t_a_i_tensor))

    def h_term(self, h_upper, h_lower):
        a_h_shift = [self.params.i if a_check == self.params.a else 0 for a_check in (h_upper, h_lower)]
        return self.tensors.h_full[a_h_shift[0]:h_upper + a_h_shift[0], a_h_shift[1]:h_lower + a_h_shift[1]]

    
    def v_term(self, v_upper_1, v_upper_2, v_lower_1, v_lower_2, v_site_1, v_site_2):
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
        # else:
        #     if self.params.gap and ((v_site_1 == self.params.gap_site and v_site_2 == self.params.gap_site + 1) or
        #                             (v_site_1 == self.params.gap_site + 1 and v_site_2 == self.params.gap_site)):
        #         return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))
        #     if abs(v_site_1 - v_site_2) == 1:
        #         a_v_shift = [self.params.i if a_check == self.params.a else 0 for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)]
        #         return self.tensors.v_full[
        #             a_v_shift[0]:v_upper_1 + a_v_shift[0],
        #             a_v_shift[1]:v_upper_2 + a_v_shift[1],
        #             a_v_shift[2]:v_lower_1 + a_v_shift[2],
        #             a_v_shift[3]:v_lower_2 + a_v_shift[3]
        #         ]
        #     return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))

    def t_term(self, t_site_1, t_site_2):
        return self.tensors.t_ab_ij_tensor[np.abs(t_site_2 - t_site_1)]

    def update_one(self, r_1_value):
        a, i, eps = self.params.a, self.params.i, self.params.epsilon
        update = np.zeros((a, i))
        for u_a in range(a):
            for u_i in range(i):
                update[u_a, u_i] = 1 / (eps[u_a + i] - eps[u_i])
        return update * r_1_value

    def update_two(self, r_2_value):
        a, i, eps = self.params.a, self.params.i, self.params.epsilon
        update = np.zeros((a, a, i, i))
        for u_a in range(a):
            for u_b in range(a):
                for u_i in range(i):
                    for u_j in range(i):
                        update[u_a, u_b, u_i, u_j] = 1 / (eps[u_a + i] + eps[u_b + i] - eps[u_i] - eps[u_j])
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
        x_s = 0

        # Unpack parameters
        site, i_method = self.params.site, self.params.i_method
        a, i, p = self.params.a, self.params.i, self.params.p
        A, B = self.terms.a_term, self.terms.b_term

        # Initialize residual
        R_single = np.zeros((a, i), dtype=complex)

        # Precompute shared terms for x_s = 0
        H_pq = self.terms.h_pp         # shape (p, p)
        
        # Term 1: A H B
        R_single += oe.contract("ap,pq,qi->ai", A, H_pq, B, optimize='optimal')

        sites_close = [1, site-1]

        # Terms from other sites
        for z_s in sites_close:
            if i_method >= 1:
                # Term 2: A V T
                V_cd = self.v_term(p, i, a, a, x_s, z_s)      # (p, l, c, d)
                T_cd = self.t_term(x_s, z_s)                  # (c, d, i, l)
                R_single += oe.contract("ap,plcd,cdil->ai", A, V_cd, T_cd, optimize='optimal')

            # Term 3: A V B B
            V_qq = self.v_term(p, i, p, p, x_s, z_s)          # shape (p, l, q, s)
            R_single += oe.contract("ap,plqs,qi,sl->ai", A, V_qq, B, B, optimize='optimal')

        return R_single


    def residual_double_sym(self, y_d: int) -> np.ndarray:
        """
        Computes symmetric double residual R^{ab}_{ij}(0, y_d) 
        assuming site x_d = 0 is fixed and y_d varies.
        Uses optimized tensor contractions via opt_einsum.
        """

        site, i_method = self.params.site, self.params.i_method
        p, i, a = self.params.p, self.params.i, self.params.a
        A, B = self.terms.a_term, self.terms.b_term
        x_d = 0  # fixed by symmetry

        R = np.zeros((a, a, i, i), dtype=complex)

        if i_method >= 1:

            # Term 1: A ⊗ A · V · B ⊗ B
            V_pqrs = self.v_term(p, p, p, p, x_d, y_d)
            R += oe.contract("ap,bq,pqrs,ri,sj->abij", A, A, V_pqrs, B, B, optimize='optimal')

            if i_method >= 2:
                T_0y = self.t_term(x_d, y_d)

                # Term 2: A ⊗ A · V · T
                V_pqcd = self.v_term(p, p, a, a, x_d, y_d)
                R += oe.contract("ap,bq,pqcd,cdij->abij", A, A, V_pqcd, T_0y, optimize='optimal')

                # Term 3: T · V · B ⊗ B
                V_klpq = self.v_term(i, i, p, p, x_d, y_d)
                R -= oe.contract("abkl,klpq,pi,qj->abij", T_0y, V_klpq, B, B, optimize='optimal')

                if i_method == 3 and site >= 4:
                    # Term 4: T · V · T
                    V_klcd = self.v_term(i, i, a, a, x_d, y_d)
                    R -= oe.contract("abkl,klcd,cdij->abij", T_0y, V_klcd, T_0y, optimize='optimal')

                    # Term 5: all connected permutations
                    for z in range(site):
                        for w in range(site):
                            if z not in {x_d, y_d} and w not in {x_d, y_d} and z != w:
                                if abs(z - w) == 1 or abs(z - w) == (self.params.site - 1):
                                    V_klcd_zw = self.v_term(i, i, a, a, z, w)
                                    T_0z = self.t_term(x_d, z)
                                    T_yw = self.t_term(y_d, w)
                                    
                                    R += oe.contract("klcd,acik,bdjl->abij", V_klcd_zw, T_0z, T_yw, optimize='optimal')

        return R

    def residual_double_non_sym_1(self, y_d: int) -> np.ndarray:
        """
        Computes asymmetric residual R^{ab}_{ij} for fixed x_d = 0 and variable y_d.
        Corresponds to the first non-symmetric contraction path using optimized einsums.
        """

        site, i_method = self.params.site, self.params.i_method
        p, i, a = self.params.p, self.params.i, self.params.a
        A, B, h_pa, h_ip = self.terms.a_term, self.terms.b_term, self.terms.h_pa, self.terms.h_ip
        x_d = 0  # fixed

        R = np.zeros((a, a, i, i), dtype=complex)

        if i_method >= 1:
            T_xy = self.t_term(x_d, y_d)

            # Term 1
            R += oe.contract("ap,pc,cbij->abij", A, h_pa, T_xy, optimize='optimal')

            # Term 2
            R -= oe.contract("abkj,kp,pi->abij", T_xy, h_ip, B, optimize='optimal')

            if i_method >= 2:

                for z in range(site):
                    if z != x_d and z != y_d:

                        # Term 3
                        V_ipap = self.v_term(i, p, a, p, z, y_d)
                        T_xz = self.t_term(x_d, z)
                        R += oe.contract("acik,krcs,br,sj->abij", T_xz, V_ipap, A, B, optimize='optimal')

                        # Term 4
                        V_piap = self.v_term(p, i, a, p, y_d, z)
                        R += oe.contract("bq,qlds,adij,sl->abij", A, V_piap, T_xy, B, optimize='optimal')

                        # Term 5
                        V_iipp = self.v_term(i, i, p, p, z, x_d)
                        R -= oe.contract("abkj,lkrp,pi,rl->abij", T_xy, V_iipp, B, B, optimize='optimal')

        return R


    def residual_double_non_sym_2(self, y_d: int) -> np.ndarray:
        """
        Computes asymmetric residual R^{ba}_{ji} for fixed x_d = 0 and variable y_d.
        Corresponds to the second non-symmetric contraction path using optimized einsums.
        """

        site, i_method = self.params.site, self.params.i_method
        p, i, a = self.params.p, self.params.i, self.params.a
        A, B, h_pa, h_ip = self.terms.a_term, self.terms.b_term, self.terms.h_pa, self.terms.h_ip
        x_d = 0  # fixed

        R = np.zeros((a, a, i, i), dtype=complex)

        if i_method >= 1:
            T_yx = self.t_term(y_d, x_d)

            # Term 1
            R += oe.contract("bp,pc,caji->baji", A, h_pa, T_yx, optimize='optimal')

            # Term 2
            R -= oe.contract("baki,kp,pj->baji", T_yx, h_ip, B, optimize='optimal')

            if i_method >= 2:

                for z in range(site):
                    if z != x_d and z != y_d:

                        # Term 3
                        V_ipap = self.v_term(i, p, a, p, z, x_d)
                        T_yz = self.t_term(y_d, z)
                        R += oe.contract("bcjk,krcs,ar,si->baji", T_yz, V_ipap, A, B, optimize='optimal')

                        # Term 4
                        V_piap = self.v_term(p, i, a, p, x_d, z)
                        R += oe.contract("aq,qlds,bdji,sl->baji", A, V_piap, T_yx, B, optimize='optimal')

                        # Term 5
                        V_iipp = self.v_term(i, i, p, p, z, y_d)
                        R -= oe.contract("baki,lkrp,pj,rl->baji", T_yx, V_iipp, B, B, optimize='optimal')

        return R


    def residual_double_total(self, y_d: int) -> np.ndarray:
        return (self.residual_double_sym(y_d) +
                self.residual_double_non_sym_1(y_d) +
                self.residual_double_non_sym_2(y_d))
    

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