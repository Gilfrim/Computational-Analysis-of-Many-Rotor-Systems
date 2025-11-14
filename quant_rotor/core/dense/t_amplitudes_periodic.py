import numpy as np

from quant_rotor.core.dense.hamiltonian_big import hamiltonian_general_dense
from quant_rotor.models.dense.support_ham import (
    basis_m_to_p_matrix_conversion,
    write_matrix_elements,
)
from quant_rotor.models.dense.t_amplitudes_sub_class import (
    QuantumSimulation,
    SimulationParams,
    TensorData,
)


def t_periodic(
    site: int,
    state: int,
    g: float,
    i_method: int = 3,
    threshold: float = 1e-8,
    gap: bool = False,
    gap_site: int = 3,
    low_state: int = 1,
    K_import: np.ndarray = [],
    V_import: np.ndarray = [],
    t_1_import: np.ndarray = [],
    t_2_import: np.ndarray = [],
    Import_K_V: bool = False,
    Import_t: bool = False,
    NO: bool = False,
    periodic: bool = True,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    site : int
        The number of rotors (sites) in the system.
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
    # state variables
    # could just use p, i, a
    # makes checking einsums and such a bit easier
    p = state
    i = low_state
    a = p - i

    if Import_K_V:

        h_full = K_import
        v_full = V_import

    elif NO:

        _, K, V = hamiltonian_general_dense(state, site, g)

        h_full = K
        v_full = V.reshape(p, p, p, p)

    else:

        K, V = write_matrix_elements((state - 1) // 2)

        V_tensor = V.reshape(p, p, p, p)  # Adjust if needed

        h_full = basis_m_to_p_matrix_conversion(K, state)
        v_full = basis_m_to_p_matrix_conversion(V_tensor, state)

        v_full = v_full * g

    if Import_t:
        t_a_i_tensor = t_1_import
        t_ab_ij_tensor = t_2_import
    else:
        t_a_i_tensor = np.full((site, a, i), 0.0, dtype=complex)
        t_ab_ij_tensor = np.full((site, site, a, a, i, i), 0.0, dtype=complex)

    # eigenvalues from h for update
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
        periodic=periodic,
    )

    tensors = TensorData(
        t_a_i_tensor=t_a_i_tensor,
        t_ab_ij_tensor=t_ab_ij_tensor,
        h_full=h_full,
        v_full=v_full,
    )

    qs = QuantumSimulation(params, tensors)

    single = np.zeros((site, a, i), dtype=complex)
    double = np.zeros((site, site, a, a, i, i), dtype=complex)

    while True:

        single[0] = qs.residual_single(0)
        for y_site in range(1, site):
            single[y_site] = single[0]
            double[0, y_site] = qs.residual_double_total(0, y_site)
            for x_site in range(1, site):
                double[x_site, (x_site + y_site) % site] = double[0, y_site]

        # for x_site in range(site):
        #     single[x_site] = qs.residual_single(x_site)
        #     for y_site in range(site):
        #         if x_site < y_site:
        #             double[x_site, y_site] = qs.residual_double_total(x_site, y_site)

        one_max = single.flat[np.argmax(np.abs(single))]
        two_max = double.flat[np.argmax(np.abs(double))]

        tensors.t_a_i_tensor[0] -= qs.update_one(single[0])

        for site_1 in range(1, site):
            tensors.t_a_i_tensor[site_1] = tensors.t_a_i_tensor[0]
            tensors.t_ab_ij_tensor[0, site_1] -= qs.update_two(double[0, site_1])
            for site_2 in range(1, site):
                tensors.t_ab_ij_tensor[site_2, (site_1 + site_2) % site] = (
                    tensors.t_ab_ij_tensor[0, site_1]
                )

        # for site_u_1 in range(site):
        #     tensors.t_a_i_tensor[site_u_1] -= qs.update_one(single[site_u_1])
        #     for site_u_2 in range(site):
        #         if site_u_1 < site_u_2:
        #             tensors.t_ab_ij_tensor[site_u_1, site_u_2] -= qs.update_two(
        #                 double[site_u_1, site_u_2]
        #             )

        if np.all(abs(single) <= threshold) and np.all(abs(double) <= threshold):
            break

        # CHANGE BACK TO 10
        if abs(one_max) >= 100 or abs(two_max) >= 100:
            raise ValueError("Diverges.")

    energy = 0

    if periodic:
        # energy calculations
        for site_x in range(site):
            energy += np.einsum("ip, pi->", qs.h_term(i, p), qs.B_term(i, site_x))

            for site_y in range(site_x + 1, site_x + site):
                if abs(site_x - site_y) == 1 or abs(site_x - site_y) == (site - 1):
                    # noinspection SpellCheckingInspection
                    energy += (
                        np.einsum(
                            "ijab, abij->",
                            qs.v_term(i, i, a, a, site_x, site_y % site),
                            qs.t_term(site_x, site_y % site),
                        )
                        * 0.5
                    )
                    # noinspection SpellCheckingInspection
                    energy += (
                        np.einsum(
                            "ijpq, pi, qj->",
                            qs.v_term(i, i, p, p, site_x, site_y % site),
                            qs.B_term(i, site_x),
                            qs.B_term(i, site_y % site),
                        )
                        * 0.5
                    )
    else:
        # energy calculations
        for site_x in range(site):
            energy += np.einsum("ip, pi->", qs.h_term(i, p), qs.B_term(i, site_x)) * 0.5

            for site_y in range(site):
                if site_x < site_y:
                    # noinspection SpellCheckingInspection
                    energy += np.einsum(
                        "ijab, abij->",
                        qs.v_term(i, i, a, a, site_x, site_y % site),
                        qs.t_term(site_x, site_y % site),
                    )
                    # noinspection SpellCheckingInspection
                    energy += np.einsum(
                        "ijpq, pi, qj->",
                        qs.v_term(i, i, p, p, site_x, site_y % site),
                        qs.B_term(i, site_x),
                        qs.B_term(i, site_y % site),
                    )

    return (
        one_max,
        two_max,
        energy,
        tensors.t_a_i_tensor,
        tensors.t_ab_ij_tensor,
    )
