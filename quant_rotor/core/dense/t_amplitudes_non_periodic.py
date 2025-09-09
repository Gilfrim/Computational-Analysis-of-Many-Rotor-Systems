import numpy as np
from quant_rotor.models.dense.support_ham import write_matrix_elements, basis_m_to_p_matrix_conversion
from quant_rotor.models.dense.t_amplitudes_sub_class import QuantumSimulation, TensorData, SimulationParams


#printout settings for large matrices
np.set_printoptions(suppress = True, linewidth = 1500, threshold = 10000, precision = 12)

def HF_test(start_point: str, g: float, tensors: TensorData ,params: SimulationParams):
        
        h_full, v_full = tensors.h_full, tensors.v_full
        
        state, i, = params.state, params.i
    #crit point negative is just for testing should change
        crit_point = -0.5
        # Initialize the density matrix based on the starting point and the matrix form
        if g > crit_point:
            # ⟨m|cosφ|n⟩ = 0.5 * δ_{m, n+1} + 0.5 * δ_{m, n-1}
            # ⟨m|sinφ|n⟩ = 1/(2i) * δ_{m, n+1} + 1/(2i) * δ_{m, n-1}
            if start_point == "sin":
                matrix = np.zeros((state, state), dtype = complex)
            else:
                matrix = np.zeros((state, state))

            #uses real values for cos and complex for sin
            t = {"cos": 0.5, "sin": 1 / 2j}[start_point]

            #adds elements to trig matrix
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
            if not np.allclose(fock_vec.conj().T @ fock_vec, np.identity(state)):
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
        if not np.allclose(fock_final_vec.conj().T @ fock_final_vec, np.identity(state)):
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

def t_non_periodic(
    site: int,
    state: int,
    g: float,
    i_method: int = 3,
    threshold: float=1e-8,
    gap: bool = False,
    gap_site: int=3,
    HF: bool=False,
    start_point: str="sin",
    low_state: int=1,
    t_a_i_tensor_initial: np.ndarray=0,
    t_ab_ij_tensor_initial: np.ndarray=0
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    site : int
        The number of rotors (site) in the system.
    state : int
        Total number states in the system, counting the ground state. Ex: system of -1, 0, 1 would be a system of 3 states.
    g_val : float
        The constant multiplier for the potential energy. Typically in the range 0 <= g <= 1.
    i_method : int, optional
        Chosing between iterative methods. , by default 3
    threshold : float, optional
        The threshold for convergence of the residuals, by default 1e-8
    gap : bool, optional
        The gap between , by default False
    gap_site : int, optional
        _description_, by default 3
    HF : bool, optional
        Chossing wether to implement or not the HF presidure. True -> implement; False -> not implement., by default False
    start_point : str, optional
        Condition used for HF presidure. , by default "sin"
    low_state : int, optional
        Defines how many ground states does particles in a system have., by default 1
    t_a_i_tensor_initial : np.ndarray, optional
        In case of input of the t_1 amplitude for propatation. , by default 0
    t_ab_ij_tensor_initial : np.ndarray, optional
        In case of input of the t_2 amplitude for propatation. , by default 0

    Returns
    -------
    tuple[float, float, float, np.ndarray, np.ndarray]
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    #state variables
    #could just use p, i, a
    #makes checking einsums and such a bit easier
    p = state
    i = low_state
    a = p - i
    periodic = False

    # Load .npy matrices directly from the package
    K, V = write_matrix_elements((state-1)//2)

    V = (V + V.T - np.diag(np.diag(V)))/2
    V_tensor = V.reshape(p, p, p, p)  # Adjust if needed

    h_full = basis_m_to_p_matrix_conversion(K, state)
    v_full = basis_m_to_p_matrix_conversion(V_tensor, state)

    v_full = v_full * g

    if np.isscalar(t_a_i_tensor_initial) and np.isscalar(t_ab_ij_tensor_initial):
        t_a_i_tensor = np.full((site, a, i), t_a_i_tensor_initial, dtype=complex)
        t_ab_ij_tensor = np.full((site, site, a, a, i, i), t_ab_ij_tensor_initial, dtype=complex)
    else:
        t_a_i_tensor = t_a_i_tensor_initial
        t_ab_ij_tensor = t_ab_ij_tensor_initial

    #eigenvalues from h for update
    epsilon = np.diag(h_full)

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
    periodic=periodic
    )

    tensors = TensorData(
    t_a_i_tensor=t_a_i_tensor,
    t_ab_ij_tensor=t_ab_ij_tensor,
    h_full=h_full,
    v_full=v_full
    )

    qs = QuantumSimulation(params, tensors)

    print(h_full)
    print(v_full)

    del K, V, h_full, v_full, t_a_i_tensor, t_ab_ij_tensor, V_tensor

    # if HF:
    #     HF_test(start_point, g, tensors, params)
    # else:

    #     params.epsilon = np.diag(tensors.h_full)

    iteration = 0

    single = np.zeros((site, a, i), dtype = complex)
    double = np.zeros((site, site, a, a, i ,i), dtype = complex)

    previous_energy = 0



    while True:

        energy = 0

        for x_site in range(site):
            single[x_site] = qs.residual_single(x_site)
            for y_site in range(site):
                if x_site < y_site:
                    double[x_site, y_site] = qs.residual_double_total(x_site, y_site)

        # if HF:
        #     #middle = np.zeros((p, q))
        #
        #     #print(f"b term\n{B_term(s, l, 0)}")
        #     x_test = 2
        #     middle = h_term(p, q, x_test)
        #     h_val, h_vec = np.linalg.eigh(middle)
        #     for z_test in range(site):
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

        print(f"1 max: {one_max}")
        print(f"2 max: {two_max}\n")

        #calculates update values for residuals
        for site_u_1 in range(site):
            tensors.t_a_i_tensor[site_u_1] -= qs.update_one(single[site_u_1])
            for site_u_2 in range(site):
                if site_u_1 < site_u_2:
                    tensors.t_ab_ij_tensor[site_u_1, site_u_2] -= qs.update_two(double[site_u_1, site_u_2])
                    #t_ab_ij_tensor[site_u_2, site_u_1] -= update_two(double[site_u_1, site_u_2])

        #energy calculations
        for site_x in range(site):
            energy += np.einsum("ip, pi->", qs.h_term(i, p), qs.B_term(i, site_x)) #* 0.5
            for site_y in range(site):
                if site_x < site_y:
                    # noinspection SpellCheckingInspection
                    energy += np.einsum("ijab, abij->", qs.v_term(i, i, a, a, site_x, site_y), qs.t_term(site_x, site_y))
                    # noinspection SpellCheckingInspection
                    energy += np.einsum("ijpq, pi, qj->", qs.v_term(i, i, p, p, site_x, site_y), qs.B_term(i, site_x), qs.B_term(i, site_y))

        if np.all(abs(single) <= threshold) and np.all(abs(double) <= threshold):
            break

        #CHANGE BACK TO 10
        if abs(one_max) >= 100 or abs(two_max) >= 100:
            raise ValueError("Diverges.")

        delta_energy = energy - previous_energy
        previous_energy = energy

        iteration += 1

    return one_max, two_max, energy, tensors.t_a_i_tensor, tensors.t_ab_ij_tensor