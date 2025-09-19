import numpy as np

from quant_rotor.models.dense.support_ham import (
    basis_m_to_p_matrix_conversion,
    write_matrix_elements,
)


def density_matrix_1(states: int, sites: int, eigvector: np.ndarray, choose_site: int) -> np.ndarray:

    # Reduced density matrix for chosen site (states x states)
    D_1 = np.zeros((states, states), dtype=complex)

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
            D_1[p, p_prime] = val

    return D_1

def density_matrix_2(states: int, sites: int, eigvector: np.ndarray, x: int, y: int) -> np.ndarray:

    if y < x:
        ValueError("y is less than x in density matrix 2")

    D_2 = np.zeros((states, states, states, states), dtype=complex)

    n_lambda = states**(x % (sites-1))
    n_mu = states**((sites - y - 1) % (sites - 1))
    n_nu = states**(np.abs(y - x) - 1)

    check_states = n_lambda*n_mu*n_nu*states**2 - states**(sites)

    if check_states != 0:
        print("The number of cases DON'T match in states:", states, "sites:", sites)

    for q in range(states):
        for q_prime in range(states):
            for p in range(states):
                for p_prime in range(states):

                    val = 0.0

                    for Lambda in range(int(n_lambda)):
                        for mu in range(int(n_mu)):
                            for nu in range(int(n_nu)):

                                i = mu + q*n_mu + nu*states*n_mu + p*n_nu*n_mu*states + Lambda*n_nu*n_mu*states**2
                                j = mu + q_prime*n_mu + nu*states*n_mu + p_prime*n_nu*n_mu*states + Lambda*n_nu*n_mu*states**2

                                val += eigvector.conj()[i] * eigvector[j]

                    D_2[p, q, p_prime, q_prime] = val

    return D_2

def dencity_energy(states: int, sites: int, g: int, psi_vec: np.ndarray)-> float:

    K, V = write_matrix_elements((states-1) // 2)

    V = V + V.T - np.diag(np.diag(V))
    V = V.reshape(states,states,states,states)
    V *= g

    K = basis_m_to_p_matrix_conversion(K)
    V = basis_m_to_p_matrix_conversion(V)

    h_D_x = 0
    V_D_xy = 0

    for x in range(sites):
        for y in range(x + 1, sites):

            D_x = density_matrix_1(states, sites, psi_vec, x)
            D_xy = density_matrix_2(states, sites, psi_vec, x, y)
            D_xy = D_xy.reshape(states, states, states, states)

            h_D_x += np.einsum("qp,qp->", K, D_x)
            V_D_xy += np.einsum("rsij,rsij->", V, D_xy)

    return h_D_x + V_D_xy
