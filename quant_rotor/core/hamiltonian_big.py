import numpy as np
from quant_rotor.core.hamiltonian import hamiltonian
from quant_rotor.models.density_matrix import density_matrix_1

def hamiltonian_big(state: int, site: int, g_val: float, H_K_V: list[np.ndarray], l_val: float=0) -> np.ndarray:

    H = H_K_V[0]
    K = H_K_V[1]
    V = H_K_V[2]

    state_old = int(K.shape[0])
    site_old = int(np.log(H.shape[0]) / np.log(state_old))

    eig_val, eig_vec = np.linalg.eig(H)

    index = np.argsort(eig_val)
    psi_vec = eig_vec[:, index[0]] 
    rho_site_0 = density_matrix_1(state_old, site_old, psi_vec, 0)
    eig_val_D, eig_vec_D = np.linalg.eig(rho_site_0)

    matrix_p_to_natural_orbital = eig_vec_D[:, 0:state]

    # print("System sites:", site)
    # print("Shape p_to_mu:", matrix_p_to_natural_orbital.shape)
    # print("Shape K:", K.shape)
    # print("Shape V:", V.shape)
    # print("Shape H:", H.shape)

    V = V.reshape(state_old**2,state_old**2)
    K_mu = matrix_p_to_natural_orbital.T.conj() @ K @ matrix_p_to_natural_orbital
    V_mu = np.kron(matrix_p_to_natural_orbital.T.conj(), matrix_p_to_natural_orbital.T.conj()) @ V @ np.kron(matrix_p_to_natural_orbital, matrix_p_to_natural_orbital)
    V_mu = V_mu.reshape(state,state,state,state)

    H_mu = hamiltonian(state, site, g_val, l_val, K_mu, V_mu, True)[0]

    return H_mu, K_mu, V_mu, matrix_p_to_natural_orbital

def hamiltonian_general(states: int, sites: int, g_val: float):

    H_K_V = hamiltonian(11, 3, g_val)
    for current_site in range(3, sites + 2, 2):
        H_K_V = hamiltonian_big(states, current_site, g_val, H_K_V)

    return H_K_V