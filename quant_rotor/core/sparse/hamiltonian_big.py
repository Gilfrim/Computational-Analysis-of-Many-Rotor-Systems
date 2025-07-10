from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix, kron
from scipy.sparse.linalg import eigsh
from quant_rotor.core.dense.hamiltonian import hamiltonian_dence
from quant_rotor.models.dense.density_matrix import density_matrix_1

def hamiltonian_big(state: int, site: int, g_val: float, H_K_V: list[np.ndarray], l_val: float=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

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

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
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

    # H_sparse_csr = H.tocsr()

    # eig_val, eig_vec = eigsh(H_sparse_csr, k=1, which='SM')

    # H = H.toarray()

    eig_val, eig_vec = np.linalg.eigh(H)

    index = np.argsort(eig_val)
    psi_vec = eig_vec[:, index[0]] 

    rho_site_0 = density_matrix_1(state_old, site_old, psi_vec, 0)

    eig_val_D, matrix_p_to_natural_orbital = np.linalg.eigh(rho_site_0)
    index_d = np.argsort(-eig_val_D)

    # matrix_p_to_natural_orbital_sparse = csr_matrix(matrix_p_to_natural_orbital[:, index_d[:state]])
    matrix_p_to_natural_orbital_sparse = matrix_p_to_natural_orbital[:, index_d[:state]]

    V = V.reshape(state_old**2,state_old**2)
    K_mu = matrix_p_to_natural_orbital_sparse.T.conj() @ K @ matrix_p_to_natural_orbital_sparse
    V_mu = np.kron(matrix_p_to_natural_orbital_sparse.T.conj(), matrix_p_to_natural_orbital_sparse.T.conj()) @ V @ np.kron(matrix_p_to_natural_orbital_sparse, matrix_p_to_natural_orbital_sparse)

    # left = kron(matrix_p_to_natural_orbital_sparse.T.conj(), matrix_p_to_natural_orbital_sparse.T.conj())
    # right = kron(matrix_p_to_natural_orbital_sparse, matrix_p_to_natural_orbital_sparse)

    # V_mu = left @ V @ right

    # V_mu = V_mu.reshape(state,state,state,state)

    H_mu = hamiltonian_dence(state, site, g_val, l_val, K_mu, V_mu, True, False, False)[0]

    return H_mu, K_mu, V_mu#, matrix_p_to_natural_orbital[:, index_d[:state]]

def hamiltonian_general(states: int, sites: int, g_val: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """    
    Parameters
    ----------
    state : int
        Number of unique states in the system, not counting the ground state. Ex: system of -1, 0, 1 would be a system of 1
    site : int
        The number of rotors in the system.
    g_val : float
        The constant multiplier for the potential energy. Usually in the range of 0 <= g <= 1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Returns a tuple of Kinetic energy matrix, Potential energy matrix and a Hamiltonian in the respective order.
        Dence Kinetic energy matrix in basis p and dimension of (state, state).
        Dence Potential energy matrix n basis p and shape of (state, state, state, state) simetric along the diagonal.
        Dence Hamiltonian of shape (state^site, state^site) constructed from above Kinetic and Potential.
    """

    H_K_V = hamiltonian_dence(11, 3, g_val, spar=False)
    for current_site in range(3, sites + 2, 2):
        H_K_V = hamiltonian_big(states, current_site, g_val, H_K_V)

    return H_K_V

def hamiltonian_big(state: int, site: int, g_val: float, H_K_V: list[np.ndarray], l_val: float=0) -> int:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    state : int
        _description_
    site : int
        _description_
    g_val : float
        _description_
    H_K_V : list[np.ndarray]
        _description_
    l_val : float, optional
        _description_, by default 0

    Returns
    -------
    int
        _description_
    """    
    return 4