import numpy as np

from quant_rotor.CCC_iterative_methods.Dense.t_amplitudes_sub_class import (
    QuantumSimulation,
    SimulationParams,
    TensorData,
)
from quant_rotor.CCC_iterative_methods.Dense.t_amplitudes_sub_class_linked import (
    QuantumSimulation_linked,
    SimulationParams_linked,
    TensorData_linked,
)
from quant_rotor.CCC_iterative_methods.Dense.t_amplitudes_sub_class_transformed import (
    QuantumSimulation_Transformed,
    SimulationParams_Transformed,
    TensorData_Transformed,
)


def t_periodic(
    site: int,
    state: int,
    h_full: np.ndarray,
    v_full_xy: np.ndarray,
    v_full_yx: np.ndarray,
    presidure_type: str,
    i_method: int = 3,
    threshold: float = 1e-10,
    gap: bool = False,
    gap_site: int = 3,
    low_state: int = 1,
    t_1_import: np.ndarray = [],
    t_2_import: np.ndarray = [],
    Import_t: bool = False,
    periodic: bool = True,
    one_cicle: bool = False,
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

    if Import_t:
        t_a_i_tensor = t_1_import
        t_ab_ij_tensor = t_2_import
    else:
        t_a_i_tensor = np.full((site, a, i), 0.0, dtype=complex)
        t_ab_ij_tensor = np.full((site, site, a, a, i, i), 0.0, dtype=complex)

    # eigenvalues from h for update
    epsilon = np.diag(h_full)

    print(presidure_type)

    if presidure_type == "original":

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
            v_full_xy=v_full_xy,
            v_full_yx=v_full_yx,
        )

        qs = QuantumSimulation(params, tensors)

    elif presidure_type == "linked":

        params = SimulationParams_linked(
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

        tensors = TensorData_linked(
            t_a_i_tensor=t_a_i_tensor,
            t_ab_ij_tensor=t_ab_ij_tensor,
            h_full=h_full,
            v_full_xy=v_full_xy,
            v_full_yx=v_full_yx,
        )

        qs = QuantumSimulation_linked(params, tensors)

    elif presidure_type == "transformed":

        params = SimulationParams_Transformed(
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

        tensors = TensorData_Transformed(
            t_a_i_tensor=t_a_i_tensor,
            t_ab_ij_tensor=t_ab_ij_tensor,
            h_full=h_full,
            v_full_xy=v_full_xy,
            v_full_yx=v_full_yx,
        )

        qs = QuantumSimulation_Transformed(params, tensors)

    single = np.zeros((site, a, i), dtype=complex)
    double = np.zeros((site, site, a, a, i, i), dtype=complex)

    counter = 0

    while True:

        for x_site in range(site):
            single[x_site] = qs.residual_single(x_site)
            for y_site in range(site):
                if x_site < y_site:
                    # print(qs.c2(x_site, y_site))
                    double[x_site, y_site] = qs.residual_double_total(x_site, y_site)

        for x_site in range(site):
            for y_site in range(site):
                if x_site < y_site:

                    double[y_site, x_site] = (
                        double[x_site, y_site].reshape(a, a).T.reshape(a, a, i, i)
                    )

        # for x_site in range(site):
        #     if not (np.array_equal(single[0], single[1])):
        #         print(f"R_1 on {x_site}: {np.max(np.abs(single[0] - single[1]))}")
        #     else:
        #         print(f"R_1 on {x_site}: {True}")
        #     for y_site in range(site):
        #         if x_site < y_site:
        #             if not (
        #                 np.array_equal(
        #                     double[x_site, y_site].reshape(a, a),
        #                     double[y_site, x_site].reshape(a, a).T,
        #                 )
        #             ):
        #                 print(
        #                     f"R_2 on {x_site, y_site}: {np.max(np.abs(double[x_site, y_site].reshape(a, a) - double[y_site, x_site].reshape(a, a).T))}"
        #                 )
        #             else:
        #                 print(f"R_2 on {x_site, y_site}: {True}")

        if one_cicle:
            return t_a_i_tensor, t_ab_ij_tensor, single, double

        one_max = single.flat[np.argmax(np.abs(single))]
        two_max = double.flat[np.argmax(np.abs(double))]

        # print("Before", one_max, two_max)

        for site_u_1 in range(site):
            tensors.t_a_i_tensor[site_u_1] -= qs.update_one(single[site_u_1])
            for site_u_2 in range(site):
                if site_u_1 < site_u_2:
                    tensors.t_ab_ij_tensor[site_u_1, site_u_2] -= qs.update_two(
                        double[site_u_1, site_u_2]
                    )
                    tensors.t_ab_ij_tensor[site_u_2, site_u_1] -= qs.update_two(
                        double[site_u_2, site_u_1]
                    )
        counter += 1

        if np.all(abs(single) <= threshold) and np.all(abs(double) <= threshold):
            break

        # CHANGE BACK TO 10
        if abs(one_max) >= 100 or abs(two_max) >= 100:
            raise ValueError("Diverges.")

    print(counter)

    energy = 0

    if presidure_type != "transformed":
        if periodic:
            energy += np.einsum("ip, pi->", qs.h_term(i, p), qs.B_term(i, 0))
            energy += (
                np.einsum(
                    "ijab, abij->",
                    qs.v_term(i, i, a, a, 0, 1),
                    qs.t_term_2(0, 1),
                )
                / 2
            )
            # noinspection SpellCheckingInspection
            energy += (
                np.einsum(
                    "ijpq, pi, qj->",
                    qs.v_term(i, i, p, p, 0, 1),
                    qs.B_term(i, 0),
                    qs.B_term(i, 1),
                )
                / 2
            )
            energy += (
                np.einsum(
                    "ijab, abij->",
                    qs.v_term(i, i, a, a, 1, 0),
                    qs.t_term_2(1, 0),
                )
                / 2
            )
            # noinspection SpellCheckingInspection
            energy += (
                np.einsum(
                    "ijpq, pi, qj->",
                    qs.v_term(i, i, p, p, 1, 0),
                    qs.B_term(i, 1),
                    qs.B_term(i, 0),
                )
                / 2
            )

            energy = energy * site
        else:
            for site_x in range(site):
                energy += np.einsum("ip, pi->", qs.h_term(i, p), qs.B_term(i, site_x))

                for site_y in range(site):
                    if site_x < site_y:
                        # if abs(site_x - site_y) == 1:
                        # noinspection SpellCheckingInspection
                        energy += np.einsum(
                            "ijab, abij->",
                            qs.v_term(i, i, a, a, site_x, site_y),
                            qs.t_term_2(site_x, site_y),
                        )
                        # noinspection SpellCheckingInspection
                        energy += np.einsum(
                            "ijpq, pi, qj->",
                            qs.v_term(i, i, p, p, site_x, site_y),
                            qs.B_term(i, site_x),
                            qs.B_term(i, site_y),
                        )
    else:
        for site_x in range(site):
            energy += qs.E_c(site_x)
            for site_y in range(site):
                if site_x != site_y:
                    energy += qs.e_bar_c(site_x, site_y) / 2

    return (
        energy,
        tensors.t_a_i_tensor,
        tensors.t_ab_ij_tensor,
        single,
        double,
    )
