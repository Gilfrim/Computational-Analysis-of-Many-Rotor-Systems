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
    t_ab_ij_array: np.ndarray
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

    def A_term(self, a_upper: int) -> np.ndarray:
        """
        Constructs the A matrix for single excitations:
        A^{a}_{p} = [ -t^{a}_{i} | I_{a x a} ]

        It horizontally stacks the negative t-amplitudes with an identity matrix
        to form the basis transformation from (i, a) to a full-particle basis.

        Parameters
        ----------
        a_upper : int
            Size of the 'a' (virtual) index space.

        Returns
        -------
        np.ndarray
            A matrix of shape (a_upper, i + a_upper), where the first block
            contains -t^{a}_{i} and the second is the identity matrix.
        """
        a, i = self.params.a, self.params.i
        t_ai = self.tensors.t_a_i_tensor.reshape(a, i)

        # Build A = [ -t_ai | I ]
        return np.hstack((-t_ai, np.identity(a_upper)))

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

    def B_term(self, b_lower: int) -> np.ndarray:
        """
        Constructs the B vector for single excitations:
        B^{p}_{i} = [ 1_i | t^{a}_{i} ]

        It concatenates an identity-like 1-vector for the occupied space (i)
        with the t-amplitudes for the virtual space (a) to represent
        the full particle excitation vector.

        Parameters
        ----------
        b_lower : int
            Size of the occupied index space.

        Returns
        -------
        np.ndarray
            A vector of shape (b_lower + a,), concatenating 1's and t^{a}_{i}.
        """
        t_ai = self.tensors.t_a_i_tensor
        ones_i = np.ones(b_lower, dtype=t_ai.dtype)

        # Build B = [ 1_i , t_ai ]
        return np.concatenate((ones_i, t_ai))

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

    def h_term(self, h_upper, h_lower):
        a_h_shift = [self.params.i if a_check == self.params.a else 0 for a_check in (h_upper, h_lower)]
        return self.tensors.h_full[a_h_shift[0]:h_upper + a_h_shift[0], a_h_shift[1]:h_lower + a_h_shift[1]]
    
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

        def shift(size): return i if size == a else 0

        u_shift = shift(h_u)
        l_shift = shift(h_l)

        row_indices = [u_shift + u for u in range(h_u)]
        col_indices = [l_shift + l for l in range(h_l)]

        return self.tensors.h_full[np.ix_(row_indices, col_indices)]

    def v_term(self, v_upper_1, v_upper_2, v_lower_1, v_lower_2):

        a_v_shift = [self.params.i if a_check == self.params.a else 0 for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)]

        return self.tensors.v_full[
            a_v_shift[0]:v_upper_1 + a_v_shift[0],
            a_v_shift[1]:v_upper_2 + a_v_shift[1],
            a_v_shift[2]:v_lower_1 + a_v_shift[2],
            a_v_shift[3]:v_lower_2 + a_v_shift[3]]

    def v_term_sparse(self, v_u1, v_u2, v_l1, v_l2) -> sp.csr_matrix:
        """
        Extracts a sub-block from the 2D sparse tensor `v_full`, representing a
        4-index tensor V^{uv}_{wx}, flattened as (uv, wx).

        Parameters
        ----------
        v_u1, v_u2, v_l1, v_l2 : int
            Dimensions of upper and lower indices for slicing.

        Returns
        -------
        sp.csr_matrix
            Sparse submatrix of shape (v_u1 * v_u2, v_l1 * v_l2)
        """

        a, i = self.params.a, self.params.i
        dim_total = a + i

        # Determine shift (offset) for each index depending on whether
        # it belongs to the 'a' space (start after 'i') or 'i' (start at 0)
        def shift(size): return i if size == a else 0

        u1_shift = shift(v_u1)
        u2_shift = shift(v_u2)
        l1_shift = shift(v_l1)
        l2_shift = shift(v_l2)

        # Construct row and column index ranges
        row_indices = [
            (u1_shift + u1) * dim_total + (u2_shift + u2)
            for u1 in range(v_u1) for u2 in range(v_u2)
        ]
        col_indices = [
            (l1_shift + l1) * dim_total + (l2_shift + l2)
            for l1 in range(v_l1) for l2 in range(v_l2)
        ]

        # Use numpy.ix_ to extract the block
        return self.tensors.v_full[np.ix_(row_indices, col_indices)]

    def t_term(self, t_site_1, t_site_2):
        return self.tensors.t_ab_ij_array[np.abs(t_site_2 - t_site_1)]

    def update_one(self, r_1_value):
        a, i, eps, fast = self.params.a, self.params.i, self.params.epsilon, self.params.fast
        update = np.zeros((a), dtype=complex)
        for u_a in range(a):
            update[u_a] = 1 / (eps[u_a + i] - eps[0])
        return update * r_1_value

    def update_two(self, r_2_value):
        a, i, eps = self.params.a, self.params.i, self.params.epsilon
        update = np.zeros((a, a), dtype=complex)
        for u_a in range(a):
            for u_b in range(a):
                        update[u_a, u_b] = 1 / (eps[u_a + i] + eps[u_b + i] - 2*eps[0])
        return update * r_2_value

    
    def update_one_sparse(self, r_1_value):
        """
        Sparse version: returns a diagonal sparse matrix containing the update
        values multiplied by r_1_value.
        """
        a, i, eps, fast = self.params.a, self.params.i, self.params.epsilon, self.params.fast
        diag_values = np.array([1 / (eps[u_a + i] - eps[0]) for u_a in range(a)], dtype=complex)
        return sp.diags(diag_values, offsets=0, format="csr") * r_1_value


    def update_two_sparse(self, r_2_value):
        """
        Sparse version: returns a CSR sparse matrix for the 2D update.
        """
        a, i, eps = self.params.a, self.params.i, self.params.epsilon
        rows, cols, data = [], [], []

        for u_a in range(a):
            for u_b in range(a):
                val = 1 / (eps[u_a + i] + eps[u_b + i] - 2 * eps[0])
                if val != 0:  # avoid storing unnecessary zeros
                    rows.append(u_a)
                    cols.append(u_b)
                    data.append(val)

        update_sparse = sp.csr_matrix((data, (rows, cols)), shape=(a, a), dtype=complex)
        return update_sparse * r_2_value

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

        if fast:
            R_single = sp.csr_matrix((a, 1), dtype=complex)
        else:
            R_single = np.zeros((a), dtype=complex)

        # Term 1: A H B
        if fast:
            H_pq = self.terms.h_pp

            R_single += (A_ap @ (H_pq @ B_q))
        else:
            H_pq = self.terms.h_pp

            R_single += oe.contract("ap,pq,q->a", A_ap, H_pq, B_q)

        # Terms from other sites
        for z_s in [1, site-1]:
            if i_method >= 1:

                V_pC = self.terms.V_piaa
                V_pQ = self.terms.V_pipp

                # Term 2: A V T
                if fast:
                    T_C = self.t_term(x_s, z_s).reshape(a**2, 1)

                    print(T_C.toarray().reshape(a**2))

                    R_single += (A_ap @ (V_pC @ T_C))
                else:
                    T_C = self.t_term(x_s, z_s).reshape(a**2)

                    R_single += oe.contract("ap,pC,C->a", A_ap, V_pC, T_C)
            # Term 3: A V B B
            if fast:
                R_single += (A_ap @ (V_pQ @ BB_Q))
            else:
                R_single += oe.contract("ap,pQ,Q->a", A_ap, V_pQ, BB_Q)
            
        print("\n")
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
        
        if fast:
            R = sp.csr_matrix((a, a), dtype=complex)
        else:
            R = np.zeros((a, a), dtype=complex)

        x_d = 0  # fixed by symmetry

        if i_method >= 1:

            # Term 1: A ⊗ A · V · B ⊗ B
            if abs(0 - y_d) == 1 or abs(0 - y_d) == (site - 1):
                if fast:
                    V_QR = self.terms.V_pppp

                    R += ((AA_BQ @ (V_QR @ BB_R)).reshape(a, a))
                else:
                    V_QR = self.terms.V_pppp

                    R += oe.contract("BQ,QR,R->B", AA_BQ, V_QR, BB_R).reshape(a, a)

            if i_method >= 2:
                if abs(0 - y_d) == 1 or abs(0 - y_d) == (site - 1):
                    if fast:
                        T_C_flat = self.t_term(x_d, y_d).reshape(a**2, 1)
                        T_C = self.t_term(x_d, y_d).reshape(a, a)
                        V_QC = self.terms.V_ppaa
                        V_pq = self.terms.V_iipp

                        R += (AA_BQ @ (V_QC @ T_C_flat)).reshape(a, a)
                        R -= (T_C * ((V_pq @ B).T @ B)[0 ,0])
                    else:
                        T_C_flat = self.t_term(x_d, y_d).reshape(a**2)
                        T_C = self.t_term(x_d, y_d)
                        V_QC = self.terms.V_ppaa
                        V_pq = self.terms.V_iipp
                        
                        # Term 2: A ⊗ A · V · T
                        R += oe.contract("BQ,QC,C->B", AA_BQ, V_QC, T_C_flat).reshape(a, a)

                        # Term 3: T · V · B ⊗ B
                        R -= oe.contract("ab,pq,p,q->ab", T_C, V_pq, B, B)

                if i_method == 3 and site >= 4:
                    if fast:
                        V_cd = self.terms.V_iiaa.reshape(a, a)

                        if abs(0 - y_d) == 1 or abs(0 - y_d) == (site - 1):
                            R -= (T_C * np.sum(V_cd.multiply(T_C)))
                    else:
                        V_cd = self.terms.V_iiaa                        
                        # Term 4: T · V · T
                        if abs(0 - y_d) == 1 or abs(0 - y_d) == (site - 1):
                            R -= oe.contract("ab,cd,cd->ab", T_C, V_cd, T_C)

                    # Term 5: all connected permutations
                    for z in range(site-1):
                            if z not in {0, y_d} and z+1 not in {0, y_d} and z != z+1:
                                    if fast:
                                        T_0z_1 = self.t_term(x_d, z)
                                        T_yw_1 = self.t_term(y_d, z+1)
                                        T_0z_2 = self.t_term(x_d, z+1)
                                        T_yw_2 = self.t_term(y_d, z)

                                        R += T_0z_1 @ V_cd @ T_yw_1.T
                                        R += T_0z_2 @ V_cd @ T_yw_2.T
                                    else:
                                        T_0z_1 = self.t_term(x_d, z)
                                        T_yw_1 = self.t_term(y_d, z+1)
                                        T_0z_2 = self.t_term(x_d, z+1)
                                        T_yw_2 = self.t_term(y_d, z)

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
        
        if fast:
            R = sp.csr_matrix((a, a), dtype=complex)
        else:
            R = np.zeros((a, a), dtype=complex)

        x_d = 0

        if i_method >= 1:
            if fast:
                T_cb = self.t_term(x_d, y_d)

                R += (A @ h_pc @ T_cb)
                R -= (T_cb * (h_p @ B)[0,0])
            else:
                T_cb = self.t_term(x_d, y_d)

                # Term 1
                R += oe.contract("ap,pc,cb->ab", A, h_pc, T_cb)
                # Term 2
                R -= oe.contract("ab,p,p->ab", T_cb, h_p, B)

            if i_method >= 2:
                if fast:
                    for z in range(site):
                        if z != x_d and z != y_d:

                            V_ipap = self.terms.V_ipap.reshape(p*a, p)
                            T_xz = self.t_term(x_d, z)
                            V_pp = self.terms.V_iipp
                            V_pap = self.terms.V_piap.reshape(p*a, p)

                            if abs(z - y_d) == 1 or abs(z - y_d) == (site - 1):

                                # Term 3
                                R += (T_xz @ (A @ (V_ipap @ B).reshape(p, a)).T)

                                # Term 4                                
                                R += (T_cb @ (A @ (V_pap @ B).reshape(p, a)).T)

                            if abs(0 - z) == 1 or abs(0 - z) == (site - 1):
                                R -= (T_cb *((V_pp @ B).T @ B)[0, 0])
                else:
                    for z in range(site):
                        if z != x_d and z != y_d:

                            V_ipap = self.terms.V_ipap
                            T_xz = self.t_term(x_d, z)
                            V_pp = self.terms.V_iipp
                            V_pap = self.terms.V_piap
                            
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