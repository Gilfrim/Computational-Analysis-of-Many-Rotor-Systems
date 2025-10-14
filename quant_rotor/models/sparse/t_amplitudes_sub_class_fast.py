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

    def A_term_sparse(self, a_upper: int) -> sp.csr_matrix:
        """
        Constructs the sparse A^{a}_{p} transformation matrix used in coupled-cluster
        single-excitation operations. The matrix is composed of a negated t^{a}_{i} block
        concatenated with an identity matrix, formatted for use with sparse backends.

        Parameters
        ----------
        a_upper : int
            The number of single-particle virtual states (dimension `a`).

        Returns
        -------
        sp.csr_matrix
            A sparse matrix of shape (a, i + a), where the left block contains
            -t^{a}_{i} and the right block is an identity matrix I^{a}_{a}.
        """
        t_ai = sp.csr_matrix(self.tensors.t_a_i_tensor.reshape(self.params.a, self.params.i))
        I = sp.identity(a_upper, format='csr', dtype=t_ai.dtype)
        return sp.hstack([-t_ai, I], format='csr')

    def B_term(self, b_lower):
        return np.concatenate((np.ones(b_lower), self.tensors.t_a_i_tensor))

    def B_term_sparse(self, b_lower: int) -> sp.csr_matrix:
        """
        Constructs the sparse B^{p}_{i} transformation vector used in coupled-cluster
        projection operations. This consists of a vertical concatenation of an all-ones
        vector and the t^{a}_{i} tensor.

        Parameters
        ----------
        b_lower : int
            The number of occupied single-particle states (dimension `i`).

        Returns
        -------
        sp.csr_matrix
            A sparse column vector of shape (i + a, 1), where the first block is filled
            with ones (for occupied modes), and the second block is t^{a}_{i}.
        """
        ones_i = sp.csr_matrix(np.ones((b_lower, 1)))
        t_ai = sp.csr_matrix(self.tensors.t_a_i_tensor.reshape(-1, 1))
        return sp.vstack([ones_i, t_ai], format='csr')

    def h_term_sparse(self, h_u: int, h_l: int) -> sp.csr_matrix:
        """
        Extracts a sub-block from the 2D sparse matrix `h_full`, representing a
        2-index tensor H^{u}_{l} or H_{uv} depending on the context.

        Parameters
        ----------
        h_u : int
            Dimension of the upper index.
        h_l : int
            Dimension of the lower index.

        Returns
        -------
        sp.csr_matrix
            Sparse submatrix of shape (h_u, h_l)
        """
        a, i = self.params.a, self.params.i
        dim_total = a + i

        def shift(size):
            return i if size == a else 0

        u_shift = shift(h_u)
        l_shift = shift(h_l)

        row_indices = [u_shift + u for u in range(h_u)]
        col_indices = [l_shift + l for l in range(h_l)]

        return self.tensors.h_full[np.ix_(row_indices, col_indices)]

    def v_term_sparse(self, v_u1: int, v_u2: int, v_l1: int, v_l2: int) -> sp.csr_matrix:
        """
        Return a sparse sub-block of V (stored as a 2D sparse matrix 'v_full')
        corresponding to the 4-index slice:
            V[(u1,u2),(l1,l2)]
        where sizes passed are the block sizes for each index (typically 'a' or 'i').

        The storage layout is assumed to be:
            rows: r = (u1_shift + u1)*D + (u2_shift + u2)
            cols: c = (l1_shift + l1)*D + (l2_shift + l2)
        with D = a + i and shifts = i if that index is in the 'a' block, else 0.
        The function returns a CSR sparse matrix of shape (v_u1*v_u2, v_l1*v_l2).
        """
        a, i = self.params.a, self.params.i
        D = a + i

        # shift(index_size) = i if that index is the 'a' block, else 0 (for 'i' block)
        def shift(sz: int) -> int:
            return i if sz == a else 0

        u1s, u2s, l1s, l2s = shift(v_u1), shift(v_u2), shift(v_l1), shift(v_l2)

        # Build row indices (not a single contiguous slice in general)
        # For each u1 in [0..v_u1-1], take the block of rows for u2 in [0..v_u2-1]
        row_idx = []
        for u1 in range(v_u1):
            base = (u1s + u1) * D + u2s
            row_idx.extend(range(base, base + v_u2))
        row_idx = np.asarray(row_idx, dtype=np.int64)

        # Build column indices similarly
        col_idx = []
        for l1 in range(v_l1):
            base = (l1s + l1) * D + l2s
            col_idx.extend(range(base, base + v_l2))
        col_idx = np.asarray(col_idx, dtype=np.int64)

        V = self.tensors.v_full  # must be scipy.sparse (CSR/CSC/COO, etc.)

        # Fancy indexing in two steps preserves sparsity:
        #  - select rows, then select columns
        sub = V[row_idx, :][:, col_idx]

        # Ensure CSR format on output
        return sub.tocsr()

    def t_term(self, t_site_1, t_site_2):
        return self.tensors.t_ab_ij_tensor[np.abs(t_site_2 - t_site_1)]

    def update_one(self, r_1_value):
        a, i, eps, fast = self.params.a, self.params.i, self.params.epsilon, self.params.fast
        update = sp.csr_matrix((a, i), dtype=complex)
        for u_a in range(a):
            update[u_a] = 1 / (eps[u_a + i] - eps[0])
        return r_1_value.multiply(update)

    def update_two(self, r_2_value):
        a, i, eps = self.params.a, self.params.i, self.params.epsilon
        update = sp.csr_matrix((a, a), dtype=complex)
        for u_a in range(a):
            for u_b in range(a):
                update[u_a, u_b] = 1 / (eps[u_a + i] + eps[u_b + i] - 2 * eps[0])
        return update.multiply(r_2_value)

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
        site, i_method, fast = self.params.site, self.params.i_method, self.params.fast
        a, i, p = self.params.a, self.params.i, self.params.p

        A_ap, B_q, BB_Q = self.terms.a_term, self.terms.b_term, self.terms.bb_term
        R_single = sp.csr_matrix((a, 1), dtype=complex)

        # Term 1: A H B
        H_pq = self.terms.h_pp

        R_single += (A_ap @ (H_pq @ B_q))

        # Terms from other sites
        for z_s in [1, site-1]:
            if i_method >= 1:

                # Term 2: A V T
                V_pC = self.terms.V_piaa
                T_C = self.t_term(x_s, z_s).reshape(a**2, 1)
                V_pQ = self.terms.V_pipp

                R_single += (A_ap @ (V_pC @ T_C))

            # Term 3: A V B B
            R_single += (A_ap @ (V_pQ @ BB_Q))

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
        R = sp.csr_matrix((a, a), dtype=complex)

        x_d = 0  # fixed by symmetry

        if i_method >= 1:

            # Term 1: A ⊗ A · V · B ⊗ B
            if abs(0 - y_d) == 1 or abs(0 - y_d) == (site - 1):
                V_QR = self.terms.V_pppp

                R += ((AA_BQ @ (V_QR @ BB_R)).reshape(a, a))

            if i_method >= 2:
                if abs(0 - y_d) == 1 or abs(0 - y_d) == (site - 1):
                    T_C = self.t_term(x_d, y_d).reshape(a, a)
                    T_C_flat = T_C.reshape(a**2, 1)
                    V_QC = self.terms.V_ppaa
                    V_pq = self.terms.V_iipp

                    R += (AA_BQ @ (V_QC @ T_C_flat)).reshape(a, a)
                    R -= (T_C * ((V_pq @ B).T @ B)[0 ,0])

                if i_method == 3:
                    V_cd = self.terms.V_iiaa.reshape(a, a)

                    if abs(0 - y_d) == 1 or abs(0 - y_d) == (site - 1):
                        R -= (T_C * np.sum(V_cd.multiply(T_C)))

                    # Term 5: all connected permutations
                    if site >= 4:
                        for z in range(site - 1):
                            if (
                                z not in {0, y_d}
                                and z + 1 not in {0, y_d}
                                and z != z + 1
                            ):
                                T_0z_1 = self.t_term(x_d, z)
                                T_yw_1 = self.t_term(y_d, z + 1)
                                T_0z_2 = self.t_term(x_d, z + 1)
                                T_yw_2 = self.t_term(y_d, z)

                                R += T_0z_1 @ V_cd @ T_yw_1.T
                                R += T_0z_2 @ V_cd @ T_yw_2.T
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
        R = sp.csr_matrix((a, a), dtype=complex)
        x_d = 0

        if i_method >= 1:
            T_cb = self.t_term(x_d, y_d)

            R += (A @ h_pc @ T_cb)
            R -= (T_cb * (h_p @ B)[0,0])

            if i_method >= 2:
                for z in range(site):
                    if z != x_d and z != y_d:

                        V_ipap = self.terms.V_ipap
                        T_xz = self.t_term(x_d, z)
                        V_pp = self.terms.V_iipp
                        V_pap = self.terms.V_piap

                        if abs(z - y_d) == 1 or abs(z - y_d) == (site - 1):

                            # Term 3
                            R += (T_xz @ (A @ (V_ipap @ B).reshape(p, a)).T)

                            # Term 4
                            R += (T_cb @ (A @ (V_pap @ B).reshape(p, a)).T)

                        if abs(0 - z) == 1 or abs(0 - z) == (site - 1):
                            R -= (T_cb *((V_pp @ B).T @ B)[0, 0])
        return R

    def residual_double_total(self, y_d: int) -> np.ndarray:
        return (self.residual_double_sym(y_d) + self.residual_double_non_sym_1(y_d)*2)
