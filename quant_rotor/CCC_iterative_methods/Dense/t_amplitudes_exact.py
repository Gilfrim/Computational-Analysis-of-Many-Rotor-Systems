import numpy as np

from quant_rotor.CCC_iterative_methods.Dense.t_amplitudes_sub_class import (
    QuantumSimulation,
    SimulationParams,
    TensorData,
)
from quant_rotor.Hamiltonian_models.Dense.support_ham import (
    basis_m_to_p_matrix_conversion,
    write_matrix_elements,
)


def intermediate_normalisation(eig_val: np.ndarray, eig_vec: np.ndarray) -> np.ndarray:

    min_index = np.argmin(eig_val)

    reference_ground_state = eig_vec[:, min_index]

    d = reference_ground_state / reference_ground_state[0]

    return d


# def t_1_amplitutde(site_a: int, state_a: int, states: int, d: np.ndarray) -> int:
#     index = state_a * states**site_a
#     return d[index]


def t_1_amplitutde(x: int, state_a: int, state: int, sites: int, d: np.ndarray) -> int:

    # Define the total number of elements in the matrix operator, which represent the left and right sites that are not interacting
    # by n_lambda and n_mu respectively.
    n_mu = state ** (sites - x - 1)

    i = state_a * n_mu

    # print(f"Site: {x}, State{state_a}: {i}")

    return d[i]


# def t_2_amplitutde(
#     site_a: int, state_a: int, site_b: int, state_b: int, states: int, d: np.ndarray
# ) -> int:

#     t_1_a = t_1_amplitutde(site_a, state_a, states, d)
#     t_1_b = t_1_amplitutde(site_b, state_b, states, d)
#     C_ab = d[state_a * states**site_a + state_b * states**site_b]

#     t_2_ab = C_ab - t_1_a * t_1_b

#     return t_2_ab, C_ab


def t_2_amplitutde(
    x: int, state_a: int, y: int, state_b: int, state: int, sites: int, d: np.ndarray
) -> int:

    t_1_a = t_1_amplitutde(x, state_a, state, sites, d)
    t_1_b = t_1_amplitutde(y, state_b, state, sites, d)

    y = (x + 1) % sites

    n_mu_x = state ** (sites - x - 1)
    n_mu_y = state ** (sites - y - 1)

    i_x = state_a * n_mu_x
    i_y = state_b * n_mu_y

    C_ab = d[i_x + i_y]

    t_2_ab = C_ab - t_1_a * t_1_b

    return t_2_ab, C_ab


# def amplitute_energy(
#     sites: int,
#     states: int,
#     g: float,
#     d: np.ndarray,
#     Import: bool = False,
#     K_import: np.ndarray = [],
#     V_import: np.ndarray = [],
# ):

#     if Import:

#         K = K_import
#         V = V_import

#     else:
#         K, V = write_matrix_elements((states - 1) // 2)

#         V = V + V.T - np.diag(np.diag(V))
#         V = V.reshape(states, states, states, states)
#         V *= g

#         K = basis_m_to_p_matrix_conversion(K, states)
#         V = basis_m_to_p_matrix_conversion(V, states)

#     E_0 = K[0, 0] * sites + np.einsum("ijij->", V) * states**sites

#     sum_t_1 = 0
#     sum_t_2 = 0

#     t_1_max = 0
#     t_2_max = 0

#     for site_a in range(sites):
#         for state_a in range(1, states):

#             t1 = t_1_amplitutde(site_a, state_a, states, sites, d)
#             sum_t_1 += K[state_a, 0] * t1

#             val_1 = np.max(np.abs(t1))

#             if val_1 > t_1_max:
#                 t_1_max = val_1

#             for site_b in range(sites):
#                 if site_a < site_b:
#                     print(site_a, site_b)
#                     for state_b in range(1, states):

#                         C2 = t_2_amplitutde(
#                             site_a, state_a, site_b, state_b, states, sites, d
#                         )[1]
#                         sum_t_2 += V[state_a, state_b, 0, 0] * C2

#                         val_2 = np.max(
#                             np.abs(
#                                 t_2_amplitutde(
#                                     site_a, state_a, site_b, state_b, states, sites, d
#                                 )[0]
#                             )
#                         )

#                         if val_2 > t_2_max:
#                             t_2_max = val_2

#     return E_0 + sum_t_1 + sum_t_2, sum_t_1, sum_t_2, E_0


def amplitute_energy(
    site: int,
    state: int,
    periodic: bool,
    K: np.ndarray,
    V_xy: np.ndarray,
    V_yx: np.ndarray,
    t_1: np.ndarray,
    t_2: np.ndarray,
):

    p = state
    i = 1
    a = p - i

    # E_0 = K[0, 0] * site + np.einsum("ijij->", V) * state**site
    epsilon = np.diag(K)

    params = SimulationParams(
        a=a,
        i=i,
        p=p,  # These can be the same as `a + i` or chosen independently
        site=site,
        state=state,
        i_method=3,
        gap=False,
        gap_site=3,
        epsilon=epsilon,
        periodic=periodic,
    )

    tensors = TensorData(
        t_a_i_tensor=t_1,
        t_ab_ij_tensor=t_2,
        h_full=K,
        v_full_xy=V_xy,
        v_full_yx=V_yx,
    )

    qs = QuantumSimulation(params, tensors)

    energy = 0

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

    return energy


def t_1_amplitude_guess_ground_state(
    states: int,
    sites: int,
    eig_vec: np.ndarray,
    eig_val: np.ndarray,
    low_states: int = 1,
):

    i = low_states
    a = states - low_states
    t_a_i_tensor = np.full((sites, a, i), 0, dtype=complex)

    d = intermediate_normalisation(eig_val, eig_vec)

    for site in range(sites):
        for state in range(states - 1):
            t_a_i_tensor[site, state, i - 1] = t_1_amplitutde(
                site, state + 1, states, sites, d
            )

    return t_a_i_tensor


def t_2_amplitude_guess_ground_state(
    states: int,
    sites: int,
    eig_vec: np.ndarray,
    eig_val: np.ndarray,
    low_states: int = 1,
):

    i = low_states
    a = states - low_states
    t_ab_ij_tensor = np.full((sites, sites, a, a, i, i), 0, dtype=complex)

    d = intermediate_normalisation(eig_val, eig_vec)

    for site_a in range(sites):
        for state_a in range(a):
            for site_b in range(sites):
                for state_b in range(a):
                    if site_a < site_b:

                        t_2_guess = t_2_amplitutde(
                            site_a, state_a + 1, site_b, state_b + 1, states, sites, d
                        )[0]

                        t_ab_ij_tensor[site_a, site_b, state_a, state_b, 0, 0] = (
                            t_2_guess
                        )
                        t_ab_ij_tensor[site_b, site_a, state_b, state_a, 0, 0] = (
                            t_2_guess
                        )

    return t_ab_ij_tensor
