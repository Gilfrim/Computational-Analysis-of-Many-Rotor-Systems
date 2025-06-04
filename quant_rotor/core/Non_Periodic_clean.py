import numpy as np
from quant_rotor.models.support_ham import write_matrix_elements, basis_m_to_p_matrix_conversion
from quant_rotor.models.non_periodic_sup_class import QuantumSimulation, TensorData, SimulationParams

def save_t_1_amplitudes(iteration, t_a_i_tensor: np.ndarray):
    np.save(f"t1_amplitudes_iter_{iteration}.npy", t_a_i_tensor)

def save_t_2_amplitudes(iteration,  t_ab_ij_tensor: np.ndarray):
    np.save(f"t2_amplitudes_iter_{iteration}.npy", t_ab_ij_tensor)

def non_periodic(
    sites: int,
    states: int,
    low_states: int,
    initial: float,
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
    - sites (int): Number of lattice sites.
    - states (int): Number of rotor states.
    - low_states (int): Cutoff for low-energy states.
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
    p = states
    i = low_states
    a = p - i

    #checks for i and a sharing same state level
    if i %2 == 0 or a %2 != 0:
        raise ValueError("Overlap between low and high stats, will cause divide by zero in denominator update")

    #for states > 11 would need to have shaeers code generate csv files with m_max > 5
    #gets h values from csv files made with Shaeer's code and puts them into a dictionary

    K, V = write_matrix_elements((states-1)//2)

    V = V + V.T - np.diag(np.diag(V))
    V_tensor = V.reshape(p, p, p, p)  # Adjust if needed

    h_full = basis_m_to_p_matrix_conversion(K)
    v_full = basis_m_to_p_matrix_conversion(V_tensor)

    v_full = v_full * g

    #t1 and t2 amplitude tensors
    t_a_i_tensor = np.full((sites, a, i), initial, dtype=np.float64)
    t_ab_ij_tensor = np.full((sites, sites, a, a, i, i), initial, dtype=np.float64)

    #eigenvalues from h for update
    epsilon = np.diag(h_full)

    params = SimulationParams(
    a=a,
    i=i,
    p=p,  # These can be the same as `a + i` or chosen independently
    sites=sites,
    states=states,
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

    #unitary transformation test
    #same as transformation test.py
    
    if transform:
        tensors.h_full, tensors.v_full = qs.transformation_test()

    file_path_energy = "energy.csv"

    with open(file_path_energy, "w", encoding="utf-8") as energy_file:
        energy_file.write("Iteration, Energy, ΔEnergy\n")

        iteration = 0
        single = np.zeros((sites, a, i))
        double = np.zeros((sites, sites, a, a, i, i))
        previous_energy = 0

        while True:
            energy = 0

            for x_site in range(sites):
                single[x_site] = qs.residual_single(x_site)
                for y_site in range(sites):
                    if x_site < y_site:
                        double[x_site, y_site] = qs.residual_double_total(x_site, y_site)

            print(f"1 max: {np.max(np.abs(single))}")
            print(f"2 max: {np.max(np.abs(double))}")

            if np.all(abs(single) <= threshold) and np.all(abs(double) <= threshold):
                break

            if np.isnan(single).any() or np.isnan(double).any() or np.isinf(single).any() or np.isinf(double).any():
                raise ValueError("Diverges (inf or nan)")

            # Save amplitudes as NumPy arrays
            # save_t_1_amplitudes(iteration, tensors.t_a_i_tensor)
            # save_t_2_amplitudes(iteration, tensors.t_ab_ij_tensor)

            for site_u_1 in range(sites):
                tensors.t_a_i_tensor[site_u_1] -= qs.update_one(single[site_u_1])
                for site_u_2 in range(sites):
                    if site_u_1 < site_u_2:
                        tensors.t_ab_ij_tensor[site_u_1, site_u_2] -= qs.update_two(double[site_u_1, site_u_2])

            for site_x in range(sites):
                energy += np.einsum("ip, pi->", qs.h_term(i, p), qs.B_term(i, site_x))
                for site_y in range(sites):
                    if site_x < site_y:
                        energy += np.einsum("ijab, abij->", qs.v_term(i, i, a, a, site_x, site_y), qs.t_term(site_x, site_y))
                        energy += np.einsum("ijpq, pi, qj->", qs.v_term(i, i, p, p, site_x, site_y), qs.B_term(i, site_x), qs.B_term(i, site_y))

            delta_energy = energy - previous_energy
            previous_energy = energy

            iteration += 1
            print(f"Iteration #: {iteration}")
            print(f"Energy: {float(energy)}")

            energy_file.write(f"{iteration}, {energy}, {delta_energy}\n")
    
    return previous_energy, tensors.t_a_i_tensor, tensors.t_ab_ij_tensor



if __name__ == "__main__":
    energy, t_a_i_tensor, t_ab_ij_tensor = non_periodic(5, 5, 1, 0, 1e-8, 0.5, 3, False, 3, False)
    print("Energy:", energy)
    print(f"1 max: {np.max(np.abs(t_a_i_tensor))}")
    print(f"2 max: {np.max(np.abs(t_ab_ij_tensor))}")