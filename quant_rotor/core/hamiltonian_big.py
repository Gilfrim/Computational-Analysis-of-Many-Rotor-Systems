import numpy as np
import quant_rotor.core.hamiltonian as h

def density_matrix(states: int, sites: int, eigvector: np.ndarray, choose_site: int) -> np.ndarray:

    # Reduced density matrix for chosen site (states x states)
    D = np.zeros((states, states), dtype=complex)

    n_lambda = states**(choose_site)
    n_mu = states**(sites - choose_site - 1)

    # Loop over the local states at the chosen site
    for p in range(states):
        for p_prime in range(states):
            val = 0.0

            # Sum over all other sites (Lambda and mu represent other sites)
            for Lambda in range(int(n_lambda)):
                for mu in range(int(n_mu)):

                    # Full indices in the Hilbert space
                    i = mu + p * n_mu + Lambda * states * n_mu
                    j = mu + p_prime * n_mu + Lambda * states * n_mu

                    # Accumulate contribution from the eigenvector
                    val += eigvector.conj()[i] * eigvector[j]

            # Set the computed value into the reduced density matrix
            D[p, p_prime] = val

    return D

def hamiltonian_big(state: int, site: int, g: float, H_K_V: list[np.ndarray]) -> np.ndarray:

    H = H_K_V[0]
    K = H_K_V[1]
    V = H_K_V[2]

    eig_val, eig_vec = np.linalg.eig(H)

    index = np.argsort(eig_val)

    psi_vec = eig_vec[:, index[0]] 
    rho_site_0 = density_matrix(11, 3, psi_vec, 0)
    eig_val_D, eig_vec_D = np.linalg.eig(rho_site_0)

    matrix_p_to_mu = eig_vec_D[:, 0:state]

    V = V.reshape(state_old**2,state_old**2)
    K_mu = matrix_p_to_mu.T.conj() @ K @ matrix_p_to_mu
    V_mu = np.kron(matrix_p_to_mu.T.conj(), matrix_p_to_mu.T.conj()) @ V @ np.kron(matrix_p_to_mu, matrix_p_to_mu)
    V_mu = V_mu.reshape(state,state,state,state)

    H_mu = h.hamiltonian(state, site, g, K_mu, V_mu, True)[0]

    return H_mu, K_mu, V_mu

if __name__ == "__main__":
    site = 5
    state = 5
    g = 0.1

    state_old = 11
    site_old = 3
    g_old = 0.1

    H_K_V = h.hamiltonian(state_old, site_old, g_old)

    print(hamiltonian_big(state, site, g, H_K_V).shape)