import numpy as np
import scipy.sparse as sp
from quant_rotor.core.sparse.hamiltonian import hamiltonian_sparse
from quant_rotor.models.dense.density_matrix import density_matrix_1

def hamiltonian_big_sparse(state: int, site: int, g_val: float, H_K_V: list[np.ndarray], tau: float, l_val: float=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """    
    Parameters
    ----------
    state : int
        Number of unique states in the system, not counting the ground state. Ex: system of -1, 0, 1 would be a system of 1
    site : int
        The number of rotors in the system.
    g_val : float
        The constant multiplier for the potential energy. Usually in the range of 0 <= g <= 1.
    H_K_V:         
        Returns a tuple of Kinetic energy matrix, Potential energy matrix and a Hamiltonian in the respective order.
        Dence Kinetic energy matrix in basis p and dimension of (state, state).
        Dence Potential energy matrix n basis p and shape of (state, state, state, state) simetric along the diagonal.
        Dence Hamiltonian of shape (state^site, state^site) constructed from above Kinetic and Potential.
    tau : float, optional
        Dipolar plains chain angle.
    l_val : float, optional
        A multiplier for the kinetic energy. Creates a tridiagonal matrix with zeros on the
        diagonal and l_val / sqrt(pi) on the off-diagonals. Defaults to 0 (no modification).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Returns a tuple of Kinetic energy matrix, Potential energy matrix and a Hamiltonian in the respective order.
        Dence Kinetic energy matrix in basis p and dimension of (state, state).
        Dence Potential energy matrix n basis p and shape of (state, state, state, state) simetric along the diagonal.
        Dence Hamiltonian of shape (state^site, state^site) constructed from above Kinetic and Potential.
    """

    H = H_K_V[0]
    K = H_K_V[1]
    V = H_K_V[2]

    state_old = int(K.shape[0])
    site_old = int(np.log(H.shape[0]) / np.log(state_old))

    _, psi_vec = sp.linalg.eigsh(H, k=1, which='SA')

    rho_site_0 = density_matrix_1(state_old, site_old, psi_vec, 0)

    eig_val_D, matrix_p_to_NO_full = np.linalg.eigh(rho_site_0)
    index_d = np.argsort(-eig_val_D)

    matrix_p_to_natural_orbital_sparse = sp.csr_matrix(matrix_p_to_NO_full[:, index_d[:state]])

    K_mu = matrix_p_to_natural_orbital_sparse.T.conj() @ K @ matrix_p_to_natural_orbital_sparse

    left = sp.kron(matrix_p_to_natural_orbital_sparse.T.conj(), matrix_p_to_natural_orbital_sparse.T.conj())
    right = sp.kron(matrix_p_to_natural_orbital_sparse, matrix_p_to_natural_orbital_sparse)

    V_mu = left @ V @ right

    H_mu = hamiltonian_sparse(state, site, g_val, l_val, K_mu, V_mu, True)[0]

    return H_mu, K_mu, V_mu, matrix_p_to_NO_full[:, index_d[:state]]

def hamiltonian_general_sparse(states: int, sites: int, g_val: float, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Iterages through hamiltonian systems using the hamiltonian_big_sparse to optimise the process of producing the hamiltonian.

    Parameters
    ----------
    state : int
        Number of unique states in the system, not counting the ground state. Ex: system of -1, 0, 1 would be a system of 1
    site : int
        The number of rotors in the system.
    g_val : float
        The constant multiplier for the potential energy. Usually in the range of 0 <= g <= 1.
    tau : float, optional
        Dipolar plains chain angle.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Returns a tuple of Kinetic energy matrix, Potential energy matrix and a Hamiltonian in the respective order.
        Dence Kinetic energy matrix in basis p and dimension of (state, state).
        Dence Potential energy matrix n basis p and shape of (state, state, state, state) simetric along the diagonal.
        Dence Hamiltonian of shape (state^site, state^site) constructed from above Kinetic and Potential.
    """

    H_K_V = hamiltonian_sparse(11, 3, g_val, tau, spar=False)
    for current_site in range(3, sites + 2, 2):
        H_K_V = hamiltonian_big_sparse(states, current_site, g_val, H_K_V, tau)

    return H_K_V