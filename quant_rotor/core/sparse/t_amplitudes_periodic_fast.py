from importlib.resources import files

import numpy as np
import opt_einsum as oe
import scipy.sparse as sp

from quant_rotor.models.sparse.support_ham import build_V_in_p
from quant_rotor.models.sparse.t_amplitudes_sub_class_fast import (
    PrecalcalculatedTerms,
    QuantumSimulation,
    SimulationParams,
    TensorData,
)

# printout settings for large matrices
np.set_printoptions(suppress = True, linewidth = 1500, threshold = 10000, precision = 12)

def HF_test(start_point: str, g: float, tensors: TensorData ,params: SimulationParams):

    h_full, v_full = tensors.h_full, tensors.v_full

    (
        state,
        i,
    ) = (
        params.state,
        params.i,
    )
    # crit point negative is just for testing should change
    crit_point = -0.5
    # Initialize the density matrix based on the starting point and the matrix form
    if g > crit_point:
        # ⟨m|cosφ|n⟩ = 0.5 * δ_{m, n+1} + 0.5 * δ_{m, n-1}
        # ⟨m|sinφ|n⟩ = 1/(2i) * δ_{m, n+1} + 1/(2i) * δ_{m, n-1}
        if start_point == "sin":
            matrix = np.zeros((state, state), dtype=complex)
        else:
            matrix = np.zeros((state, state))

        # uses real values for cos and complex for sin
        t = {"cos": 0.5, "sin": 1 / 2j}[start_point]

        # adds elements to trig matrix
        for off_diag in range(state - 1):
            matrix[off_diag, off_diag + 1] = t
            matrix[off_diag + 1, off_diag] = np.conj(t)

        # ⟨m|cos^2φ|n⟩ = 0.5 * δ_{m, n} + 0.5 * (δ_{m, n+2} + δ_{m, n-2})
        # ⟨m|cos^2φ|n⟩ = 0.5 * δ_{m, n} - 0.5 * (δ_{m, n+2} + δ_{m, n-2})
        I = np.identity(state)
        cos_2phi = np.zeros((state, state))

        for off in range(state - 2):
            cos_2phi[off, off + 2] = 1
            cos_2phi[off + 2, off] = 1

        # cos^2φ = (1 + cos(2φ))/ 2
        # sin^2φ = (1 - cos(2φ))/ 2
        if start_point == "cos":
            matrix_squared = 0.5 * (I + cos_2phi)
        elif start_point == "sin":
            matrix_squared = 0.5 * (I - cos_2phi)

    else:
        # uses h as starting point if g < crit point
        matrix = h_full

    # stuff I never got to
    # mix_factor = -0.1
    # h_full += mix_factor * matrix

    vals, vecs = np.linalg.eigh(matrix)

    # occupied orbitals
    U_occ = vecs[:, :i]
    density = U_occ @ U_occ.conj().T

    iteration_density = 0
    # hartree fock iterative calculations
    while True:
        iteration_density += 1
        # could use 2 * np.einsum("mknl, lk ->mn", v_full, density) instead, gives same thing
        # noinspection SpellCheckingInspection
        fock = (
            h_full
            + np.einsum("mknl, lk ->mn", v_full, density)
            + np.einsum("mknl, mn ->lk", v_full, density)
        )
        # makes sure fock is hermitian
        fock = 0.5 * fock + 0.5 * fock.conj().T

        fock_val, fock_vec = np.linalg.eigh(fock)

        # checks that fock eigenvectors are unitary
        if not np.allclose(fock_vec.conj().T @ fock_vec, np.identity(state)):
            raise ValueError("Fock vectors not unitary")

        print(f"Iteration density: {iteration_density}")

        # calculates initial fock occupied orbital
        if iteration_density == 1:
            f_occ = fock_vec[:, :i]
            f_occ_zero = f_occ.copy()
        else:
            # overlap integral
            overlap = abs(np.einsum("ka, kl->la", f_occ_zero, fock_vec))
            overlap_index = np.argmax(abs(overlap))

            # changes largest overlap eigenvector and eigenvalue to be first
            if overlap_index != 0:
                # switches the largest overlap vector to 0th position
                fock_val[[0, overlap_index]] = fock_val[[overlap_index, 0]]
                fock_vec[:, [0, overlap_index]] = fock_vec[:, [overlap_index, 0]]

            f_occ = fock_vec[:, :i]

        # calculates new density matrix
        density_update = np.einsum("ki, li ->kl", f_occ, f_occ)
        max_change = np.max(np.abs(density - density_update))
        print(f"max change: {max_change}")

        # how much the density matrix gets updated
        mix_param = 0.5
        density = mix_param * density + (1 - mix_param) * density_update

        if g > crit_point:
            # calculates the expectation value and variance of cosφ
            # haven't tested for sinφ, might need some changes
            expval = np.trace(matrix @ density)
            expval_squared = np.trace(matrix_squared @ density)

            print(f"⟨{start_point}⟩ = {np.real(expval)}")

            variance = expval_squared - expval ** 2

            print(f"Var({start_point}) = {np.real(variance)}\n")

        # stopping condition for fock matrix updates
        if max_change < 1e-10 or iteration_density == 250:
            break

    # once the density matrix has converged calculate fock lat time
    fock_final = h_full + 2 * np.einsum("mknl, lk ->mn", v_full, density)

    fock_final = 0.5 * fock_final + 0.5 * fock_final.conj().T
    fock_final_val, fock_final_vec = np.linalg.eigh(fock_final)

    # overlap integrals
    overlap_final = abs(np.einsum("ka, kl->la", f_occ_zero, fock_final_vec))
    overlap_index_final = np.argmax(abs(overlap_final))

    if overlap_index_final != 0:
        fock_final_val[[0, overlap_index_final]] = fock_final_val[[overlap_index_final, 0]]
        fock_final_vec[:, [0, overlap_index_final]] = fock_final_vec[:, [overlap_index_final, 0]]

    f_final_occ = fock_final_vec[:, :i]

    # check eigenvectors unitary
    if not np.allclose(fock_final_vec.conj().T @ fock_final_vec, np.identity(state)):
        raise ValueError("Fock final vectors not unitary")

    # checks that the difference between density and 2nd last density are the same
    density_final = np.einsum("pi, qi ->pq", f_final_occ, f_final_occ)
    if not np.allclose(density_final,density):
        print(density)
        print(density_final)
        raise ValueError("Density problem")

    # various tests to check that basis transformations are working
    # h_pq = h_full.copy()
    # v_pqrs = v_full.copy()

    # Need these
    # transformation to fock basis
    tensors.h_full = fock_final_vec.conj().T @ h_full @ fock_final_vec
    tensors.v_full = np.einsum("pi, qj, pqrs, rk, sl->ijkl", fock_final_vec, fock_final_vec, v_full, fock_final_vec, fock_final_vec)

    params.epsilon = fock_final_val


def t_periodic(
    site: int,
    state: int,
    g: float,
    fast: bool,
    i_method: int = 3,
    threshold: float = 1e-8,
    gap: bool = False,
    gap_site: int = 3,
    HF: bool = False,
    start_point: str = "sin",
    low_state: int = 1,
    t_a_i_tensor_initial: np.ndarray = 0,
    t_ab_ij_tensor_initial: np.ndarray = 0,
):
    """
    Create SimulationParams from raw input arguments.
    Performs logic for a, i, p and validates start_point.
    """
    # state variables
    # could just use p, i, a
    # makes checking einsums and such a bit easier
    p = state
    i = low_state
    a = p - i
    periodic = True

    # Load .npy matrices directly from the package
    h_full, v_full = build_V_in_p(state)

    v_full = v_full*g

    if np.isscalar(t_a_i_tensor_initial) and np.isscalar(t_ab_ij_tensor_initial):
        t_a_i_tensor = sp.csr_matrix((a, 1), dtype=complex)
        t_ab_ij_tensor = np.array([sp.csr_matrix((a, a), dtype=float) for _ in range(site)], dtype=object)

        # t_a_i_tensor = np.full((a), 0, dtype=complex)
        # t_ab_ij_tensor = np.full((site, a, a), 0, dtype=complex)
    else:
        t_a_i_tensor = t_a_i_tensor_initial
        t_ab_ij_tensor = t_ab_ij_tensor_initial

    if HF:
        HF_test(start_point, g, tensors, params)
    else:
        epsilon = h_full.diagonal()

    params = SimulationParams(
    a=a,
    i=i,
    p=p,  # These can be the same as `a + i` or chosen independently
    site=site,
    state=state,
    i_method=i_method,
    gap=gap,
    gap_site=gap_site,
    epsilon=epsilon,
    periodic=periodic,
    fast=fast
    )

    tensors = TensorData(
    t_a_i_tensor=t_a_i_tensor,
    t_ab_ij_tensor=t_ab_ij_tensor,
    h_full=h_full,
    v_full=v_full
    )

    terms = PrecalcalculatedTerms()

    qs = QuantumSimulation(params, tensors, terms)

    del h_full, v_full, t_a_i_tensor, t_ab_ij_tensor

    iteration = 0

    single = sp.csr_matrix((a, 1), dtype=complex)
    double = np.array([sp.csr_matrix((a, a), dtype=float) for _ in range(site)], dtype=object)

    # single = np.full((a), 0, dtype=complex)
    # double = np.zeros((site, a, a), dtype = complex)

    terms.h_pp=qs.h_term_sparse(p, p)
    terms.h_pa=qs.h_term_sparse(p, a)
    terms.h_ip=qs.h_term_sparse(i, p)
    terms.V_pppp=qs.v_term_sparse(p, p, p, p)
    terms.V_ppaa=qs.v_term_sparse(p, p, a, a)
    terms.V_iipp=qs.v_term_sparse(i, i, p, p).reshape(p, p)
    terms.V_iiaa=qs.v_term_sparse(i, i, a, a).reshape(a, a)
    terms.V_piaa=qs.v_term_sparse(p, i, a, a).reshape(p, a**2)
    terms.V_pipp=qs.v_term_sparse(p, i, p, p).reshape(p, p**2)
    terms.V_ipap=qs.v_term_sparse(i, p, a, p).reshape(p*a, p)
    terms.V_piap=qs.v_term_sparse(p, i, a, p).reshape(p*a, p)
    print("Done.")

    while True:

        # terms.a_term=qs.A_term(a)
        # terms.b_term=qs.B_term(i)
        # terms.bb_term=sp.csr_matrix(oe.contract("q,s->qs", terms.b_term, terms.b_term).reshape(p**2)).T
        # terms.aa_term=sp.csr_matrix(oe.contract("ap,bq->abpq", terms.a_term, terms.a_term).reshape(a**2, p**2))

        # terms.a_term=sp.csr_matrix(qs.A_term(a))
        # terms.b_term=sp.csr_matrix(qs.B_term(i)).T

        terms.a_term=qs.A_term_sparse(a)
        terms.b_term=qs.B_term_sparse(i)
        terms.bb_term=(terms.b_term @ terms.b_term.T).reshape(p**2, 1)
        terms.aa_term=sp.kron(terms.a_term, terms.a_term, format="csr")

        # print(f"a: {np.allclose(terms.a_term.toarray(), qs.A_term_sparse(a).toarray(), 1e-10)}")
        # print(f"aa: {np.allclose(terms.aa_term.toarray(), sp.kron(terms.a_term, terms.a_term, format="csr").toarray(), 1e-10)}")
        # print(f"b: {np.allclose(terms.b_term.toarray(), qs.B_term_sparse(i).toarray(), 1e-10)}")
        # print(f"bb: {np.allclose(terms.bb_term.toarray(), (terms.b_term @ terms.b_term.T).reshape(p**2, 1).toarray(), 1e-10)}")

        single = qs.residual_single()
        for y_site in range(1, site):
            double[y_site] = qs.residual_double_total(y_site)

        # one_max = single.flat[np.argmax(np.abs(single))]
        # two_max = double.flat[np.argmax(np.abs(double))]

        one_max = single.toarray().flat[np.argmax(np.abs(single.toarray()))]
        two_max = 0
        for do in double:
            if two_max < do.toarray().flat[np.argmax(np.abs(do.toarray()))]:
                two_max = do.toarray().flat[np.argmax(np.abs(do.toarray()))]

        tensors.t_a_i_tensor -= qs.update_one(single)

        for site_1 in range(1, site):
            tensors.t_ab_ij_tensor[site_1] -= qs.update_two(double[site_1])

        # if np.all(abs(single) <= threshold) and np.all(abs(double) <= threshold):
        #     break

        if np.all(abs(single.toarray()) <= threshold) and np.all(abs(np.array([mat.toarray() for mat in double])) <= threshold):
            break

        # CHANGE BACK TO 10
        if abs(one_max) >= 100 or abs(two_max) >= 100:
            raise ValueError("Diverges.")

        iteration += 1

    energy = 0

    for site_x in range(site):
        energy += (qs.terms.h_ip @ qs.terms.b_term)[0, 0]

        for site_y in range(site_x + 1, site_x + site):
            if abs(site_x - site_y) == 1 or abs(site_x - site_y) == (site - 1):
                V_iipp = qs.terms.V_iipp
                V_iiaa = qs.terms.V_iiaa
                T_xy = qs.t_term(site_x, site_y)

                # noinspection SpellCheckingInspection
                energy +=  (np.sum(V_iiaa.multiply(T_xy)) * 0.5)

                # noinspection SpellCheckingInspection
                energy += (((V_iipp @ qs.terms.b_term).T @ qs.terms.b_term) * 0.5)[0,0]

    return one_max, two_max, energy, tensors.t_a_i_tensor, tensors.t_ab_ij_tensor
