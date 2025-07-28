import numpy as np
from quant_rotor.models.dense.support_ham import write_matrix_elements, basis_m_to_p_matrix_conversion

def intermediate_normalisation(eig_val: np.ndarray, eig_vec: np.ndarray)->np.ndarray:

    min_index = np.argmin(eig_val)

    reference_ground_state = eig_vec[:, min_index]

    d = reference_ground_state / reference_ground_state[0]

    return d

def t_1_amplitutde(site_a: int, state_a: int, states: int, d: np.ndarray)-> int:
    index = state_a*states**site_a
    return d[index]

def t_2_amplitutde(site_a: int, state_a: int, site_b: int, state_b: int, states: int, d: np.ndarray)-> int:

    t_1_a = t_1_amplitutde(site_a, state_a, states, d)
    t_1_b = t_1_amplitutde(site_b, state_b, states, d)
    C_ab = d[state_a*states**site_a + state_b*states**site_b]

    t_2_ab = C_ab - t_1_a*t_1_b

    return t_2_ab, C_ab

def amplitute_energy(sites: int, states: int, g: float, d: np.ndarray):

    K, V = write_matrix_elements((states-1) // 2)

    V = V + V.T - np.diag(np.diag(V))
    V = V.reshape(states,states,states,states)
    V *= g 

    K = basis_m_to_p_matrix_conversion(K)
    V = basis_m_to_p_matrix_conversion(V)

    E_0 = K[0,0]*sites + np.einsum("ijij->", V) * states**sites

    sum_t_1 = 0
    sum_t_2 = 0

    for site_a in range(sites):
        for state_a in range(1, states):

            t1 = t_1_amplitutde(site_a, state_a, states, d)
            sum_t_1 += K[state_a, 0] * t1

            for site_b in range(site_a + 1, sites):
                for state_b in range(1, states):
                    
                    C2 = t_2_amplitutde(site_a, state_a, site_b, state_b, states, d)[1]
                    sum_t_2 += V[state_a, state_b, 0, 0] * C2

    return E_0 + sum_t_1 + sum_t_2

def t_1_amplitude_guess_ground_state(states: int, sites: int, g: float, eig_vec: np.ndarray, eig_val: np.ndarray, low_states: int=1):

    i = low_states
    a = states - low_states
    t_a_i_tensor = np.full((sites, a, i), 0, dtype=complex)

    d = intermediate_normalisation(eig_val, eig_vec)

    for site in range(sites):
        for state in range(states - 1):
            t_a_i_tensor[site, state, i-1] = t_1_amplitutde(site, state + 1, states, d)

    return t_a_i_tensor

def t_2_amplitude_guess_ground_state(states: int, sites: int, g: float,eig_vec: np.ndarray, eig_val: np.ndarray,  low_states: int=1):

    i = low_states
    a = states - low_states
    t_ab_ij_tensor = np.full((sites, sites, a, a, i, i), 0, dtype=complex)

    d = intermediate_normalisation(eig_val, eig_vec)

    for site_a in range(sites):
        for state_a in range(a):
            for site_b in range(site_a + 1, sites):
                for state_b in range(a):

                    t_2_guess = t_2_amplitutde(site_a, state_a + 1, site_b, state_b + 1, states, d)[0]

                    t_ab_ij_tensor[site_a, site_b, state_a, state_b, 0, 0] = t_2_guess
                    t_ab_ij_tensor[site_b, site_a, state_b, state_a, 0, 0] = t_2_guess

    return t_ab_ij_tensor