from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
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
    periodic: bool=True
    fast: bool=False

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
    aa_term: np.ndarray = None
    bb_term: np.ndarray = None
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
        return np.hstack((-self.tensors.t_a_i_tensor.reshape(self.params.a, self.params.i), np.identity(a_upper)))

    def B_term(self, b_lower):
        return np.concatenate((np.ones(b_lower), self.tensors.t_a_i_tensor))

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
        else:
            if self.params.gap and ((v_site_1 == self.params.gap_site and v_site_2 == self.params.gap_site + 1) or
                                    (v_site_1 == self.params.gap_site + 1 and v_site_2 == self.params.gap_site)):
                return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))
            if abs(v_site_1 - v_site_2) == 1:
                a_v_shift = [self.params.i if a_check == self.params.a else 0 for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)]
                return self.tensors.v_full[
                    a_v_shift[0]:v_upper_1 + a_v_shift[0],
                    a_v_shift[1]:v_upper_2 + a_v_shift[1],
                    a_v_shift[2]:v_lower_1 + a_v_shift[2],
                    a_v_shift[3]:v_lower_2 + a_v_shift[3]
                ]
            return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))

    def t_term(self, t_site_1, t_site_2):
        return self.tensors.t_ab_ij_tensor[np.abs(t_site_2 - t_site_1)]

    def update_one(self, r_1_value):
        a, i, eps = self.params.a, self.params.i, self.params.epsilon
        update = np.zeros((a, i))
        for u_a in range(a):
            for u_i in range(i):
                update[u_a, u_i] = 1 / (eps[u_a + i] - eps[u_i])
        return update.reshape(a) * r_1_value

    def update_two(self, r_2_value):
        a, i, eps = self.params.a, self.params.i, self.params.epsilon
        update = np.zeros((a, a, i, i))
        for u_a in range(a):
            for u_b in range(a):
                for u_i in range(i):
                    for u_j in range(i):
                        update[u_a, u_b, u_i, u_j] = 1 / (eps[u_a + i] + eps[u_b + i] - eps[u_i] - eps[u_j])
        return update.reshape(a, a) * r_2_value

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
        site, i_method, fast = self.params.site, self.params.i_method, False#self.params.fast
        a, i, p = self.params.a, self.params.i, self.params.p

        A_ap, B_q, BB_Q = self.terms.a_term, self.terms.b_term, self.terms.bb_term
        R_single = np.zeros((a), dtype=complex)

        H_pq = self.terms.h_pp
        V_pQ = self.terms.V_pipp
        V_pC = self.terms.V_piaa

        # Term 1: A H B
        if fast:
            R_single += (A_ap @ (H_pq @ B_q))
        else:
            R_single += oe.contract("ap,pq,q->a", A_ap, H_pq, B_q)

        # Terms from other sites
        for z_s in [1, site-1]:
            if i_method >= 1:
                T_C = self.t_term(x_s, z_s).reshape(a**2)
                # Term 2: A V T
                if fast:
                    R_single += (A_ap @ (V_pC @ T_C))
                else:
                    R_single += oe.contract("ap,pC,C->a", A_ap, V_pC, T_C)
                
            # Term 3: A V B B
            if fast:
                R_single += (A_ap @ (V_pQ @ BB_Q))
            else:
                R_single += oe.contract("ap,pQ,Q->a", A_ap, V_pQ, BB_Q)

        return R_single 


    def residual_double_sym(self, y_d: int) -> np.ndarray:
        """
        Computes symmetric double residual R^{ab}_{ij}(0, y_d) 
        assuming site x_d = 0 is fixed and y_d varies.
        Uses optimized tensor contractions via opt_einsum.
        """

        site, i_method, fast = self.params.site, self.params.i_method, self.params.fast
        p, i, a = self.params.p, self.params.i, self.params.a
        A, B, AA_BQ, BB_R= self.terms.a_term, self.terms.b_term, self.terms.aa_term, self.terms.bb_term
        R = np.zeros((a, a), dtype=complex)

        x_d = 0  # fixed by symmetry

        if i_method >= 1:
            V_QR = self.terms.V_pppp
            # Term 1: A ⊗ A · V · B ⊗ B
            if abs(0 - y_d) == 1 or abs(0 - y_d) == (site - 1):
                if fast:
                    R += (AA_BQ @ (V_QR @ BB_R)).reshape(a, a)
                else:
                    R += oe.contract("BQ,QR,R->B", AA_BQ, V_QR, BB_R).reshape(a, a)

            if i_method >= 2:
                T_C_flat = self.t_term(x_d, y_d).reshape(a**2)
                T_C = self.t_term(x_d, y_d)
                V_QC = self.terms.V_ppaa
                V_pq = self.terms.V_iipp

                if fast:
                    if abs(0 - y_d) == 1 or abs(0 - y_d) == (site - 1):
                        R += (AA_BQ @ (V_QC @ T_C_flat)).reshape(a, a)
                        R -= T_C * ((V_pq @ B) @ B)
                else:

                    # Term 2: A ⊗ A · V · T
                    if abs(0 - y_d) == 1 or abs(0 - y_d) == (site - 1):
                        
                        R += oe.contract("BQ,QC,C->B", AA_BQ, V_QC, T_C_flat).reshape(a, a)

                    # Term 3: T · V · B ⊗ B
                        R -= oe.contract("ab,pq,p,q->ab", T_C, V_pq, B, B)

                if i_method == 3 and site >= 4:
                    V_cd = self.terms.V_iiaa.reshape(a, a)
                    if fast:
                        if abs(0 - y_d) == 1 or abs(0 - y_d) == (site - 1):
                            scalar = np.sum(V_cd * T_C)
                            R -= T_C * scalar
                    else:        
                        # Term 4: T · V · T
                        if abs(0 - y_d) == 1 or abs(0 - y_d) == (site - 1):
                            R -= oe.contract("ab,cd,cd->ab", T_C, V_cd, T_C)
                            
                    # Term 5: all connected permutations
                    for z in range(site-1):
                            if z not in {0, y_d} and z+1 not in {0, y_d} and z != z+1:
                                    T_0z_1 = self.t_term(x_d, z)
                                    T_yw_1 = self.t_term(y_d, z+1)
                                    T_0z_2 = self.t_term(x_d, z+1)
                                    T_yw_2 = self.t_term(y_d, z)

                                    if fast:
                                        R += T_0z_1 @ V_cd @ T_yw_1.T
                                        R += T_0z_2 @ V_cd @ T_yw_2.T
                                    else:
                                        R += oe.contract("cd,ac,bd->ab", V_cd, T_0z_1, T_yw_1)
                                        R += oe.contract("cd,ac,bd->ab", V_cd, T_0z_2, T_yw_2)
        return R

    def residual_double_non_sym_1(self, y_d: int) -> np.ndarray:
        """
        Computes asymmetric residual R^{ab}_{ij} for fixed x_d = 0 and variable y_d.
        Corresponds to the first non-symmetric contraction path using optimized einsums.
        """

        site, i_method, fast = self.params.site, self.params.i_method, self.params.fast
        p, i, a = self.params.p, self.params.i, self.params.a

        A, B = self.terms.a_term, self.terms.b_term
        h_pc, h_p = self.terms.h_pa, self.terms.h_ip
        x_d = 0  # fixed

        R = np.zeros((a, a), dtype=complex)

        if i_method >= 1:
            T_cb = self.t_term(x_d, y_d)
            if fast:
                R += (A @ h_pc @ T_cb)        
                # R -= T_cb * (h_p @ B)
            else:
                # Term 1
                R += oe.contract("ap,pc,cb->ab", A, h_pc, T_cb)
                # Term 2
                R -= oe.contract("ab,p,p->ab", T_cb, h_p, B)

            if i_method >= 2:
                for z in range(site):
                    if z != x_d and z != y_d:
                        V_ipap = self.terms.V_ipap
                        T_xz = self.t_term(x_d, z)
                        V_pp = self.terms.V_iipp
                        V_pap = self.terms.V_piap
                        if fast:

                            if abs(z - y_d) == 1 or abs(z - y_d) == (site - 1):

                                # Term 3
                                R += T_xz @ (A @ (V_ipap @ B).reshape(p, a)).T

                                # Term 4
                                R += T_cb @ (A @ (V_pap @ B).reshape(p, a)).T

                            if abs(0 - z) == 1 or abs(0 - z) == (site - 1):
                                R -= T_cb *((V_pp @ B) @ B)
                        else:
                            if abs(z - y_d) == 1 or abs(z - y_d) == (site - 1):

                                # Term 3
                                R += oe.contract("ac,rcs,br,s->ab", T_xz, V_ipap, A, B)

                                # Term 4                                
                                R += oe.contract("bq,qds,ad,s->ab", A, V_pap, T_cb, B)

                            if abs(0 - z) == 1 or abs(0 - z) == (site - 1):

                                # Term 5
                                R -= oe.contract("ab,rp,p,r->ab", T_cb, V_pp, B, B)

        return R

    def residual_double_total(self, y_d: int) -> np.ndarray:
        return (self.residual_double_sym(y_d) + self.residual_double_non_sym_1(y_d)*2)
    

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