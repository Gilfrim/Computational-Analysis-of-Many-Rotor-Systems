import numpy as np
import quant_rotor.core.hamiltonian as h
import quant_rotor.core.Non_Periodic_clean as NP
from importlib.resources import files
from dataclasses import dataclass

#printout settings for large matrices
np.set_printoptions(suppress = True, linewidth = 1500, threshold = 10000, precision = 12)

@dataclass
class SimulationParams:
    a: int
    i: int
    p: int
    g: float
    sites: int
    states: int
    i_method: int
    HF: bool
    start_point: str
    epsilon: np.ndarray
    gap: bool=False
    gap_site: int=3

@dataclass
class TensorData:
    t_a_i_tensor: np.ndarray
    t_ab_ij_tensor: np.ndarray
    h_full: np.ndarray
    v_full: np.ndarray

def HF_test(tensors: TensorData ,params: SimulationParams):
        
        h_full, v_full = tensors.h_full, tensors.v_full
        
        states, start_point, i, g = params.states, params.start_point, params.i, params.g
    #crit point negative is just for testing should change
        crit_point = -0.5
        # Initialize the density matrix based on the starting point and the matrix form
        if g > crit_point:
            # ⟨m|cosφ|n⟩ = 0.5 * δ_{m, n+1} + 0.5 * δ_{m, n-1}
            # ⟨m|sinφ|n⟩ = 1/(2i) * δ_{m, n+1} + 1/(2i) * δ_{m, n-1}
            if start_point == "sin":
                matrix = np.zeros((states, states), dtype = complex)
            else:
                matrix = np.zeros((states, states))

            #uses real values for cos and complex for sin
            t = {"cos": 0.5, "sin": 1 / 2j}[start_point]

            #adds elements to trig matrix
            for off_diag in range(states - 1):
                matrix[off_diag, off_diag + 1] = t
                matrix[off_diag + 1, off_diag] = np.conj(t)


            # ⟨m|cos^2φ|n⟩ = 0.5 * δ_{m, n} + 0.5 * (δ_{m, n+2} + δ_{m, n-2})
            # ⟨m|cos^2φ|n⟩ = 0.5 * δ_{m, n} - 0.5 * (δ_{m, n+2} + δ_{m, n-2})
            I = np.identity(states)
            cos_2phi = np.zeros((states, states))

            for off in range(states - 2):
                cos_2phi[off, off + 2] = 1
                cos_2phi[off + 2, off] = 1

            # cos^2φ = (1 + cos(2φ))/ 2
            # sin^2φ = (1 - cos(2φ))/ 2
            if start_point == "cos":
                matrix_squared = 0.5 * (I + cos_2phi)
            elif start_point == "sin":
                matrix_squared = 0.5 * (I - cos_2phi)

        else:
            #uses h as starting point if g < crit point
            matrix = h_full

        #stuff I never got to
        #mix_factor = -0.1
        #h_full += mix_factor * matrix

        vals, vecs = np.linalg.eigh(matrix)

        #occupied orbitals
        U_occ = vecs[:, :i]
        density = U_occ @ U_occ.conj().T

        iteration_density = 0
        #hartree fock iterative calculations
        while True:
            iteration_density += 1
            #could use 2 * np.einsum("mknl, lk ->mn", v_full, density) instead, gives same thing
            # noinspection SpellCheckingInspection
            fock = h_full + np.einsum("mknl, lk ->mn", v_full, density) + np.einsum("mknl, mn ->lk", v_full, density)
            #makes sure fock is hermitian
            fock = 0.5 * fock + 0.5 * fock.conj().T

            fock_val, fock_vec = np.linalg.eigh(fock)

            #checks that fock eigenvectors are unitary
            if not np.allclose(fock_vec.conj().T @ fock_vec, np.identity(states)):
                raise ValueError("Fock vectors not unitary")

            print(f"Iteration density: {iteration_density}")

            #calculates initial fock occupied orbital
            if iteration_density == 1:
                f_occ = fock_vec[:, :i]
                f_occ_zero = f_occ.copy()
            else:
                #overlap integral
                overlap = abs(np.einsum("ka, kl->la", f_occ_zero, fock_vec))
                overlap_index = np.argmax(abs(overlap))

                #changes largest overlap eigenvector and eigenvalue to be first
                if overlap_index != 0:
                    #switches the largest overlap vector to 0th position
                    fock_val[[0, overlap_index]] = fock_val[[overlap_index, 0]]
                    fock_vec[:, [0, overlap_index]] = fock_vec[:, [overlap_index, 0]]

                f_occ = fock_vec[:, :i]

            #calculates new density matrix
            density_update = np.einsum("ki, li ->kl", f_occ, f_occ)
            max_change = np.max(np.abs(density - density_update))
            print(f"max change: {max_change}")

            #how much the density matrix gets updated
            mix_param = 0.5
            density = mix_param * density + (1 - mix_param) * density_update

            if g > crit_point:
                #calculates the expectation value and variance of cosφ
                #haven't tested for sinφ, might need some changes
                expval = np.trace(matrix @ density)
                expval_squared = np.trace(matrix_squared @ density)

                print(f"⟨{start_point}⟩ = {np.real(expval)}")

                variance = expval_squared - expval ** 2

                print(f"Var({start_point}) = {np.real(variance)}\n")

            #stopping condition for fock matrix updates
            if max_change < 1e-10 or iteration_density == 250:
                break

        #once the density matrix has converged calculate fock lat time
        fock_final = h_full + 2 * np.einsum("mknl, lk ->mn", v_full, density)

        fock_final = 0.5 * fock_final + 0.5 * fock_final.conj().T
        fock_final_val, fock_final_vec = np.linalg.eigh(fock_final)

        #overlap integrals
        overlap_final = abs(np.einsum("ka, kl->la", f_occ_zero, fock_final_vec))
        overlap_index_final = np.argmax(abs(overlap_final))

        if overlap_index_final != 0:
            fock_final_val[[0, overlap_index_final]] = fock_final_val[[overlap_index_final, 0]]
            fock_final_vec[:, [0, overlap_index_final]] = fock_final_vec[:, [overlap_index_final, 0]]

        f_final_occ = fock_final_vec[:, :i]

        #check eigenvectors unitary
        if not np.allclose(fock_final_vec.conj().T @ fock_final_vec, np.identity(states)):
            raise ValueError("Fock final vectors not unitary")

        #checks that the difference between density and 2nd last density are the same
        density_final = np.einsum("pi, qi ->pq", f_final_occ, f_final_occ)
        if not np.allclose(density_final,density):
            print(density)
            print(density_final)
            raise ValueError("Density problem")

        #various tests to check that basis transformations are working
        #h_pq = h_full.copy()
        #v_pqrs = v_full.copy()

        #Need these
        #transformation to fock basis
        tensors.h_full = fock_final_vec.conj().T @ h_full @ fock_final_vec
        tensors.v_full = np.einsum("pi, qj, pqrs, rk, sl->ijkl", fock_final_vec, fock_final_vec, v_full, fock_final_vec, fock_final_vec)

        params.epsilon = fock_final_val

def A_term(a_upper, a_site, tensors: TensorData):

    return np.hstack((-tensors.t_a_i_tensor[a_site], np.identity(a_upper)))

def B_term(b_lower, b_site, tensors: TensorData):
    # print(tensors.t_a_i_tensor[b_site])
    return np.vstack((np.identity(b_lower), tensors.t_a_i_tensor[b_site]))

def h_term(h_upper, h_lower, tensors: TensorData, params: SimulationParams):
    a_h_shift = [params.i if a_check == params.a else 0 for a_check in (h_upper, h_lower)]
    return tensors.h_full[a_h_shift[0]:h_upper + a_h_shift[0], a_h_shift[1]:h_lower + a_h_shift[1]]

def v_term(v_upper_1, v_upper_2, v_lower_1, v_lower_2, v_site_1, v_site_2, tensors: TensorData, params: SimulationParams):

    if abs(v_site_1 - v_site_2) == 1:
        a_v_shift = [params.i if a_check == params.a else 0 for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)]
        return tensors.v_full[
            a_v_shift[0]:v_upper_1 + a_v_shift[0],
            a_v_shift[1]:v_upper_2 + a_v_shift[1],
            a_v_shift[2]:v_lower_1 + a_v_shift[2],
            a_v_shift[3]:v_lower_2 + a_v_shift[3]
        ]
    else:
        return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))

def t_term(t_site_1, t_site_2, tensors: TensorData):
    return tensors.t_ab_ij_tensor[t_site_1, t_site_2]

def residual_single(x_s:int, tensors: TensorData, params: SimulationParams)->np.array:
    """Calculates R^{a}_{i}(x) singles equation"""

    sites, i_method, a, i, p= params.sites, params.i_method, params.a, params.i, params.p

    if params.HF and params.start_point == "sin":
        R_single = np.zeros((a, i), dtype = complex)
    else:
        R_single = np.zeros((a, i))

    R_single += np.einsum("ap, pq, qi->ai", A_term(a, x_s, tensors), h_term(p, p, tensors, params), B_term(i, x_s, tensors))
    for z_s in range(sites):
        if z_s != x_s:
            if i_method >= 1:
                # noinspection SpellCheckingInspection
                R_single += np.einsum("ap, plcd, cdil->ai", A_term(a, x_s, tensors), v_term(p, i, a, a, x_s, z_s, tensors, params), t_term(x_s, z_s, tensors))
            # noinspection SpellCheckingInspection
            R_single += np.einsum("ap, plqs, qi, sl->ai", A_term(a, x_s, tensors), v_term(p, i, p, p, x_s, z_s, tensors, params), B_term(i, x_s, tensors), B_term(i, z_s, tensors))

    return R_single

def residual_double_sym(x_d:int, y_d:int, tensors: TensorData, params: SimulationParams)->np.array:
    """Calculates Rs^{ab}_{ij}(x < y) symmetric doubles equation"""
    sites, i_method, p, i, a = params.sites, params.i_method, params.p, params.i, params.a

    if params.HF and params.start_point == "sin":
        R_double_symmetric = np.zeros((a, a, i, i), dtype = complex)
    else:
        R_double_symmetric = np.zeros((a, a, i, i))

    if i_method >= 1:
        # noinspection SpellCheckingInspection
        R_double_symmetric += np.einsum("ap, bq, pqrs, ri, sj->abij", A_term(a, x_d, tensors), A_term(a, y_d, tensors), v_term(p, p, p, p, x_d, y_d, tensors, params), B_term(i, x_d, tensors), B_term(i, y_d, tensors))
        if i_method >= 2:
            # noinspection SpellCheckingInspection
            R_double_symmetric += np.einsum("ap, bq, pqcd, cdij->abij", A_term(a, x_d, tensors), A_term(a, x_d, tensors), v_term(p, p, a, a, x_d, y_d, tensors, params), t_term(x_d, y_d, tensors))
            # noinspection SpellCheckingInspection
            R_double_symmetric -= np.einsum("abkl, klpq, pi, qj->abij", t_term(x_d, y_d, tensors), v_term(i, i, p, p, x_d, y_d, tensors, params), B_term(i, x_d, tensors), B_term(i, y_d, tensors))
            if i_method == 3:
                # noinspection SpellCheckingInspection
                R_double_symmetric -= np.einsum("abkl, klcd, cdij->abij", t_term(x_d, y_d, tensors), v_term(i, i, a, a, x_d, y_d, tensors, params), t_term(x_d, y_d, tensors))
                if sites >= 4:
                    for z_ds in range(sites):
                        for w_ds in range(sites):
                            if z_ds not in {x_d, y_d} and w_ds not in {x_d, y_d} and z_ds != w_ds:
                                # noinspection SpellCheckingInspection
                                R_double_symmetric += np.einsum("klcd, acik, bdjl->abij", v_term(i, i, a, a, z_ds, w_ds, tensors, params), t_term(x_d, z_ds, tensors), t_term(y_d, w_ds, tensors))

    return R_double_symmetric

def residual_double_non_sym_1(x_d:int, y_d:int, tensors: TensorData, params: SimulationParams)->np.array:
    """Calculates Rn^{ab}_{ij}(x, y) non-symmetric doubles equation"""

    sites, i_method, p, i, a = params.sites, params.i_method, params.p, params.i, params.a

    if params.HF and params.start_point == "sin":
        R_double_non_symmetric_1 = np.zeros((a, a, i, i), dtype = complex)
    else:
        R_double_non_symmetric_1 = np.zeros((a, a, i, i))

    if i_method >= 1:
        # noinspection SpellCheckingInspection
        R_double_non_symmetric_1 += np.einsum("ap, pc, cbij->abij", A_term(a, x_d, tensors), h_term(p, a, tensors, params), t_term(x_d, y_d, tensors))
        # noinspection SpellCheckingInspection
        R_double_non_symmetric_1 -= np.einsum("abkj, kp, pi->abij", t_term(x_d, y_d, tensors), h_term(i, p, tensors, params), B_term(i, x_d, tensors))

        if i_method >= 2:
            for z_dns_1 in range(sites):
                if z_dns_1 != x_d and z_dns_1 != y_d:
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_1 += np.einsum("acik, krcs, br, sj->abij", t_term(x_d, z_dns_1, tensors), v_term(i, p, a, p, z_dns_1, y_d, tensors, params), A_term(a, y_d, tensors), B_term(i, y_d, tensors))
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_1 += np.einsum("bq, qlds, adij, sl->abij", A_term(a, y_d, tensors), v_term(p, i, a, p, y_d, z_dns_1, tensors, params), t_term(x_d, y_d, tensors), B_term(i, z_dns_1, tensors))
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_1 -= np.einsum("abkj, lkrp, pi, rl->abij", t_term(x_d, y_d, tensors), v_term(i, i, p, p, z_dns_1, x_d, tensors, params), B_term(i, x_d, tensors), B_term(i, z_dns_1, tensors))

    return R_double_non_symmetric_1

def residual_double_non_sym_2(x_d:int, y_d:int, tensors: TensorData, params: SimulationParams)->np.array:
    """Calculates Rn^{ba}_{ji}(y, x) ai -> jb permutation non-symmetric doubles equation"""

    sites, i_method, p, i, a = params.sites, params.i_method, params.p, params.i, params.a

    if params.HF and params.start_point == "sin":
        R_double_non_symmetric_2 = np.zeros((a, a, i, i), dtype=complex)
    else:
        R_double_non_symmetric_2 = np.zeros((a, a, i, i))

    if i_method >= 1:
        # noinspection SpellCheckingInspection
        R_double_non_symmetric_2 += np.einsum("bp, pc, caji->baji", A_term(a, y_d, tensors), h_term(p, a, tensors, params), t_term(y_d, x_d, tensors))
        # noinspection SpellCheckingInspection
        R_double_non_symmetric_2 -= np.einsum("baki, kp, pj->baji", t_term(y_d, x_d, tensors), h_term(i, p, tensors, params), B_term(i, y_d, tensors))

        if i_method >= 2:
            for z_dns_2 in range(sites):
                if z_dns_2 != x_d and z_dns_2 != y_d:
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_2 += np.einsum("bcjk, krcs, ar, si->baji", t_term(y_d, z_dns_2, tensors), v_term(i, p, a, p, z_dns_2, x_d, tensors, params), A_term(a, x_d, tensors), B_term(i, x_d, tensors))
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_2 += np.einsum("aq, qlds, bdji, sl->baji", A_term(a, x_d, tensors), v_term(p, i, a, p, x_d, z_dns_2, tensors, params), t_term(y_d, x_d, tensors), B_term(i ,z_dns_2, tensors))
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_2 -= np.einsum("baki, lkrp, pj, rl->baji", t_term(y_d, x_d, tensors), v_term(i, i, p, p, z_dns_2, y_d, tensors, params), B_term(i, y_d, tensors), B_term(i ,z_dns_2, tensors))

    return R_double_non_symmetric_2

def residual_double_total(x_d:int, y_d:int, tensors: TensorData, params: SimulationParams)->np.array:
    """Calculates Rt^{ab}_{ij}(x < y) = Rs^{ab}_{ij}(x < y) + Rn^{ab}_{ij}(x < y) + Rn^{ba}_{ji}(y < x)total doubles equation"""

    return residual_double_sym(x_d, y_d, tensors, params) + residual_double_non_sym_1(x_d, y_d, tensors, params) + residual_double_non_sym_2(x_d, y_d, tensors, params)

def update_one(r_1_value, params: SimulationParams):
    a, i, eps = params.a, params.i, params.epsilon

    update = np.zeros((a, i))
    for u_a in range(a):
        for u_i in range(i):
            update[u_a, u_i] = 1 / (eps[u_a + i] - eps[u_i])
    return np.multiply(update, r_1_value)

def update_two(r_2_value, params: SimulationParams):
    a, i, eps = params.a, params.i, params.epsilon
    update = np.zeros((a, a, i, i))
    for u_a in range(a):
        for u_b in range(a):
            for u_i in range(i):
                for u_j in range(i):
                    update[u_a, u_b, u_i, u_j] = 1 / (eps[u_a + i] + eps[u_b + i] - eps[u_i] - eps[u_j])
    return np.multiply(update, r_2_value)

def create_simulation_params(
    sites: int,
    states: int,
    low_states: int,
    initial: float,
    threshold: float,
    g: float,
    i_method: int,
    HF: bool,
    start_point: str
):
    """
    Create SimulationParams from raw input arguments.
    Performs logic for a, i, p and validates start_point.
    """
    #state variables
    #could just use p, i, a
    #makes checking einsums and such a bit easier
    p = states
    i = low_states
    a = p - i

    # Load .npy matrices directly from the package
    data_dir = files("quant_rotor.data")
    K = np.load(data_dir / "K_matrix.npy")
    V = np.load(data_dir / "V_matrix.npy")

    V = V + V.T - np.diag(np.diag(V))
    V_tensor = V.reshape(p, p, p, p)  # Adjust if needed

    h_full = h.basis_m_to_p_matrix_conversion(K)
    v_full = h.basis_m_to_p_matrix_conversion(V_tensor)

    v_full = v_full * g

        #starting point using sin needs complex values
    if HF and start_point == "sin":
        t_a_i_tensor = np.full((sites, a, i), initial, dtype=complex)
        t_ab_ij_tensor = np.full((sites, sites, a, a, i, i), initial, dtype=complex)
    #otherwise use real matrices
    else:
        #t1 and t2 amplitude tensors
        t_a_i_tensor = np.full((sites, a, i), initial, dtype=np.float64)
        t_ab_ij_tensor = np.full((sites, sites, a, a, i, i), initial, dtype=np.float64)

    #eigenvalues from h for update
    epsilon = np.diag(h_full)

    params = SimulationParams(
    a=a,
    i=i,
    p=p,  # These can be the same as `a + i` or chosen independently
    g=g,
    sites=sites,
    states=states,
    i_method=i_method,
    HF=HF,
    start_point=start_point,
    epsilon=epsilon
    )

    tensors = TensorData(
    t_a_i_tensor=t_a_i_tensor,
    t_ab_ij_tensor=t_ab_ij_tensor,
    h_full=h_full,
    v_full=v_full
    )

    del K, V, h_full, v_full, t_a_i_tensor, t_ab_ij_tensor, V_tensor

    if HF:
        HF_test(tensors, params)
    else:

        params.epsilon = np.diag(tensors.h_full)

    file_path_energy = "energy.csv"

    with open(file_path_energy, "w", encoding = "utf-8") as energy_file:
        energy_file.write("Iteration, Energy, ΔEnergy\n")

        iteration = 0
        if HF and start_point == "sin":
            single = np.zeros((sites, a, i), dtype = complex)
            double = np.zeros((sites, sites, a, a, i ,i), dtype = complex)
        else:
            single = np.zeros((sites, a, i))
            double = np.zeros((sites, sites, a, a, i, i))
        previous_energy = 0

        # print(A_term(params.a, 3, tensors))
        # print(B_term(params.i, 3, tensors))
        print(h_term(i, i, tensors, params))

        while True:

            energy = 0

            single[0] = residual_single(0, tensors, params)
            for y_site in range(1, sites):
                single[y_site] = single[0]
                double[0, y_site] = residual_double_total(0, y_site, tensors, params)
                for x_site in range(1, sites):
                    double[x_site, (x_site + y_site) % sites] = double[0, y_site]

            # if HF:
            #     #middle = np.zeros((p, q))
            #
            #     #print(f"b term\n{B_term(s, l, 0)}")
            #     x_test = 2
            #     middle = h_term(p, q, x_test)
            #     h_val, h_vec = np.linalg.eigh(middle)
            #     for z_test in range(sites):
            #         if z_test != x_test:
            #             middle += np.einsum("plqs, sl ->pq", v_term(p, l, q, s, x_test, z_test), B_term(s, l, z_test))
            #     test = np.einsum("ap, pq, qi ->ai", A_term(a, p, x_test), middle, B_term(q, i, x_test))
            #     #print(f"middle\n {np.real(middle)}")
            #     middle_2 = h_term(p, q, x_test) + 2 * np.einsum("plqs, sl ->pq", v_term(p, l, q, s, x_test, 1), d_ij)
                #print(f"middle 2\n {np.real(middle_2)}")
                #print(f"v aux - middle\n{np.real(v_aux_2 - middle)}")
                #if iteration <= 1:
                    #print(f"fock test: {np.allclose(middle, fock_final)}")
                    #print(f"{fock_final - middle}")
                #print(f"single test: {np.allclose(test, single[x_test])}")
                #print(f"single test\n{np.real(test - single[x_test])}")

            one_max = single.flat[np.argmax(np.abs(single))]
            two_max = double.flat[np.argmax(np.abs(double))]

            # print(f"1 max: {one_max}")
            # print(f"2 max: {two_max}")

            if iteration >= 3:
                break

            if np.all(abs(single) <= threshold) and np.all(abs(double) <= threshold):
                print("I quit.")
                break

            #CHANGE BACK TO 10
            if abs(one_max) >= 100 or abs(two_max) >= 100:
                raise ValueError("Diverges")

            tensors.t_a_i_tensor[0] -= update_one(single[0], params)

            for site_1 in range(1, sites):
                tensors.t_a_i_tensor[site_1] = tensors.t_a_i_tensor[0]
                tensors.t_ab_ij_tensor[0, site_1] -= update_two(double[0, site_1], params)
                for site_2 in range(1, sites):
                    tensors.t_ab_ij_tensor[site_2, (site_1 + site_2) % sites] = tensors.t_ab_ij_tensor[0, site_1]

            #energy calculations
            for site_x in range(sites):
                energy += np.einsum("ip, pi->", h_term(i, p, tensors, params), B_term(i, site_x, tensors)) * 0.5

                for site_y in range(site_x + 1, site_x + sites):
                    # noinspection SpellCheckingInspection
                    energy += np.einsum("ijab, abij->", v_term(i, i, a, a, site_x, site_y % sites, tensors, params), t_term(site_x, site_y % sites, tensors)) * 0.5
                    # noinspection SpellCheckingInspection
                    energy += np.einsum("ijpq, pi, qj->", v_term(i, i, p, p, site_x, site_y % sites, tensors, params), B_term(i, site_x, tensors), B_term(i, site_y % sites, tensors)) * 0.5

            delta_energy = energy - previous_energy
            previous_energy = energy

            iteration += 1
            # print(f"Iteration #: {iteration}")
            # print(f"Energy: {np.real(energy)}\n")

            energy_file.write(f"{iteration}, {energy}, {delta_energy}\n")

if __name__ == "__main__":

    create_simulation_params(5, 5, 1, 0, 1e-8, 0.5, 3, False, "sin")
    print("Hello?")