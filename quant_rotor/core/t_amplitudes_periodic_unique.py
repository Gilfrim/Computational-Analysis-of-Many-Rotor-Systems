import numpy as np
from importlib.resources import files
from quant_rotor.models.support_ham import write_matrix_elements, basis_m_to_p_matrix_conversion
from quant_rotor.models.t_amplitudes_sub_class import QuantumSimulation, TensorData, SimulationParams


#printout settings for large matrices
np.set_printoptions(suppress = True, linewidth = 1500, threshold = 10000, precision = 12)

def t_periodic(
    sites: int,
    states: int,
    g: float,
    i_method: int = 3,
    threshold: float=1e-8,
    gap: bool = False,
    gap_site: int=3,
    HF: bool=False,
    start_point: str="sin",
    low_states: int=1,
    t_a_i_tensor_initial: np.ndarray=0,
    t_ab_ij_tensor_initial: np.ndarray=0
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
    periodic = True

    # Load .npy matrices directly from the package
    K, V = write_matrix_elements((states-1)//2)

    V = V + V.T - np.diag(np.diag(V))
    V_tensor = V.reshape(p, p, p, p)  # Adjust if needed

    h_full = basis_m_to_p_matrix_conversion(K)
    v_full = basis_m_to_p_matrix_conversion(V_tensor)

    v_full = v_full * g

    # if t_a_i_tensor_initial == 0 and  t_ab_ij_tensor_initial == 0:
    #     t_a_i_tensor = np.full((sites, a, i), t_a_i_tensor_initial, dtype=complex)
    #     t_ab_ij_tensor = np.full((sites, sites, a, a, i, i), t_ab_ij_tensor_initial, dtype=complex)
    # else:
    t_a_i_tensor = np.full((sites, a, i), t_a_i_tensor_initial, dtype=complex)
    t_ab_ij_tensor = np.full((sites, sites, a, a, i, i), t_ab_ij_tensor_initial, dtype=complex)


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

    del K, V, h_full, v_full, t_a_i_tensor, t_ab_ij_tensor, V_tensor

    params.epsilon = np.diag(tensors.h_full)

    iteration = 0

    single = np.zeros((sites, a, i), dtype = complex)
    double = np.zeros((sites, sites, a, a, i ,i), dtype = complex)

    previous_energy = 0

    while True:

        energy = 0

        single[0] = qs.residual_single(0)
        for y_site in range(1, sites):
            single[y_site] = single[0]
            double[0, y_site] = qs.residual_double_total(0, y_site)
            for x_site in range(1, sites):
                double[x_site, (x_site + y_site) % sites] = double[0, y_site]

        one_max = single.flat[np.argmax(np.abs(single))]
        two_max = double.flat[np.argmax(np.abs(double))]

        # print(f"1 max: {one_max}")
        # print(f"2 max: {two_max}")

        if np.all(abs(single) <= threshold) and np.all(abs(double) <= threshold):
            break

        #CHANGE BACK TO 10
        if abs(one_max) >= 100 or abs(two_max) >= 100:
            return previous_energy, tensors.t_a_i_tensor, tensors.t_ab_ij_tensor, False

        tensors.t_a_i_tensor[0] -= qs.update_one(single[0])

        for site_1 in range(1, sites):
            tensors.t_a_i_tensor[site_1] = tensors.t_a_i_tensor[0]
            tensors.t_ab_ij_tensor[0, site_1] -= qs.update_two(double[0, site_1])
            for site_2 in range(1, sites):
                tensors.t_ab_ij_tensor[site_2, (site_1 + site_2) % sites] = tensors.t_ab_ij_tensor[0, site_1]

        #energy calculations
        for site_x in range(sites):
            energy += np.einsum("ip, pi->", qs.h_term(i, p), qs.B_term(i, site_x)) * 0.5

            for site_y in range(site_x + 1, site_x + sites):
                # noinspection SpellCheckingInspection
                energy += np.einsum("ijab, abij->", qs.v_term(i, i, a, a, site_x, site_y % sites), qs.t_term(site_x, site_y % sites)) * 0.5
                # noinspection SpellCheckingInspection
                energy += np.einsum("ijpq, pi, qj->", qs.v_term(i, i, p, p, site_x, site_y % sites), qs.B_term(i, site_x), qs.B_term(i, site_y % sites)) * 0.5

        delta_energy = energy - previous_energy
        previous_energy = energy

        iteration += 1
        # print(f"Iteration #: {iteration}")
        # print(f"Energy: {np.real(energy)}\n")

            # energy_file.write(f"{iteration}, {energy}, {delta_energy}\n")
    return previous_energy, tensors.t_a_i_tensor, tensors.t_ab_ij_tensor, True

# if __name__ == "__main__":

#     t_periodic(5, 5, 1, 0, 0, 1e-8, 0.5, 3,  False, 3, True, "sin")
#     print("Hello?")