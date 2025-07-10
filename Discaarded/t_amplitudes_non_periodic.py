import numpy as np
from quant_rotor.models.dense.support_ham import write_matrix_elements, basis_m_to_p_matrix_conversion
from quant_rotor.models.dense.t_amplitudes_sub_class import QuantumSimulation, TensorData, SimulationParams

def save_t_1_amplitudes(iteration, t_a_i_tensor: np.ndarray):
    np.save(f"t1_amplitudes_iter_{iteration}.npy", t_a_i_tensor)

def save_t_2_amplitudes(iteration,  t_ab_ij_tensor: np.ndarray):
    np.save(f"t2_amplitudes_iter_{iteration}.npy", t_ab_ij_tensor)

def t_non_periodic(site: int,
    state: int,
    low_states: int,
    t_a_i_tensor_initial: np.ndarray,
    t_ab_ij_tensor_initial: np.ndarray,
    threshold: float,
    g: float,
    i_method: int,
    gap: bool,
    gap_site: int,
    transform: bool
):
    """
    Direct input version of the simulation setup.

    Parameters:
    - site (int): Number of lattice site.
    - state (int): Number of rotor state.
    - low_state (int): Cutoff for low-energy state.
    - initial (float): Initial value for the solver.
    - threshold (float): Convergence threshold.
    - g (float): Coupling constant.
    - i_method (int): Method selection index.
    - gap (bool): Whether to compute energy gap.
    - gap_site (int): Lattice site index for gap measurement (will be decremented).
    - transform (bool): Whether to apply a transformation.

    Returns:
    - dict: Placeholder output dictionary. Replace with your simulation return.
    """

    gap_site = gap_site - 1  # adjust index if needed

    #state variables
    #could just use p, i, a
    #makes checking einsums and such a bit easier
    p = state
    i = low_states
    a = p - i

    #checks for i and a sharing same state level
    if i %2 == 0 or a %2 != 0:
        raise ValueError("Overlap between low and high stats, will cause divide by zero in denominator update")

    #for state > 11 would need to have shaeers code generate csv files with m_max > 5
    #gets h values from csv files made with Shaeer's code and puts them into a dictionary

    K, V = write_matrix_elements((state-1)//2)

    V = V + V.T - np.diag(np.diag(V))
    V_tensor = V.reshape(p, p, p, p)  # Adjust if needed

    h_full = basis_m_to_p_matrix_conversion(K)
    v_full = basis_m_to_p_matrix_conversion(V_tensor)

    v_full = v_full * g

    #t1 and t2 amplitude tensors
    # t_a_i_tensor = np.full((site, a, i), t_a_i_tensor_initial, dtype=np.float64)
    # t_ab_ij_tensor = np.full((site, site, a, a, i, i), t_ab_ij_tensor_initial, dtype=np.float64)

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
    epsilon=epsilon
    )

    tensors = TensorData(
    t_a_i_tensor=t_a_i_tensor,
    t_ab_ij_tensor=t_ab_ij_tensor,
    h_full=h_full,
    v_full=v_full
    )

    qs = QuantumSimulation(params, tensors)

    del K, V, h_full, v_full, t_a_i_tensor, t_ab_ij_tensor, V_tensor
    
    if transform:
        tensors.h_full, tensors.v_full = qs.transformation_test()

    file_path_energy = "energy.csv"

    with open(file_path_energy, "w", encoding="utf-8") as energy_file:
        energy_file.write("Iteration, Energy, ΔEnergy\n")

        single = np.zeros((site, a, i), dtype = complex)
        double = np.zeros((site, site, a, a, i, i), dtype = complex)

        while True:
            energy = 0

            for x_site in range(site):
                single[x_site] = qs.residual_single(x_site)
                for y_site in range(site):
                    if x_site < y_site:
                        double[x_site, y_site] = qs.residual_double_total(x_site, y_site)

            if np.all(abs(single) <= threshold) and np.all(abs(double) <= threshold):
                break

            if np.isnan(single).any() or np.isnan(double).any() or np.isinf(single).any() or np.isinf(double).any():
                raise ValueError("Diverges (inf or nan)")

            for site_u_1 in range(site):
                tensors.t_a_i_tensor[site_u_1] -= qs.update_one(single[site_u_1])
                for site_u_2 in range(site):
                    if site_u_1 < site_u_2:
                        tensors.t_ab_ij_tensor[site_u_1, site_u_2] -= qs.update_two(double[site_u_1, site_u_2])

            for site_x in range(site):
                energy += np.einsum("ip, pi->", qs.h_term(i, p), qs.B_term(i, site_x))
                for site_y in range(site):
                    if site_x < site_y:
                        energy += np.einsum("ijab, abij->", qs.v_term(i, i, a, a, site_x, site_y), qs.t_term(site_x, site_y))
                        energy += np.einsum("ijpq, pi, qj->", qs.v_term(i, i, p, p, site_x, site_y), qs.B_term(i, site_x), qs.B_term(i, site_y))

    return energy, tensors.t_a_i_tensor, tensors.t_ab_ij_tensor