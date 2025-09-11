import numpy as np
import scipy.sparse as sp
from quant_rotor.core.dense.hamiltonian import hamiltonian_dense
from quant_rotor.models.dense.density_matrix import density_matrix_1

def hamiltonian_big_dense(state: int, site: int, g_val: float, H_K_V: tuple[np.ndarray, np.ndarray, np.ndarray], tau:float, l_val: float=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Takes in a system of rotors and scales it to a specified larger system. The approximation is taken from assuming the ground state of the 
    original system is a good description, since the majority of energy is concentrated in the ground state. Constructing and diagonalizing
    a density matrix of the ground state gives a basis for a bigger system. 

    The number of states in the new system should be less than or equal to the number of states in the old system.

    Parameters
    ----------
    state : int
        Total number states in the system, counting the ground state. Ex: system of -1, 0, 1 would be a system of 3 states.
    site : int
        The number of rotors in the new scaled system.
    g_val : float
        The constant multiplier for the potential energy. Usually in the range of 0 <= g <= 1. Should be consistant throughout the systems.
    H_K_V : tuple[np.ndarray, np.ndarray, np.ndarray]        
        Takes in a tuple with Hamiltonian matrix, Kinetic and Potential energy matricies from the original system that needs to be scased.
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
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Returns a tuple of a Hamiltonian, Kinetic and Potential energy matrix in the respective order.
        Dence Hamiltonian of shape (state^site, state^site) constructed from above Kinetic and Potential.
        Dence Kinetic energy matrix in basis p and dimension of (state, state).
        Dence Potential energy matrix n basis p and shape of (state, state, state, state) simetric along the diagonal.
    """
    #Extract the Hamiltonian, Kinetic and Potential energy matricies from a tuple.
    H = H_K_V[0]
    K = H_K_V[1]
    V = H_K_V[2]

    #Read the site and state of the original system.
    state_old = int(K.shape[0])
    site_old = int(np.log(H.shape[0]) / np.log(state_old))

    # Extract eigenstate and eigenvectors from the original system hamiltonian.
    eig_val, eig_vec = np.linalg.eigh(H)

    # Find the index associated with the smallest eigenstate.
    index = np.argmin(eig_val)
    # Extract an associated ground state eigenvector.
    ground_state_vec = eig_vec[:, index] 

    # Make a one site dencity matrix associated with the ground state.
    ground_state_dencity_matrix = density_matrix_1(state_old, site_old, ground_state_vec, 0)

    # Extract eigenstates and eigenvalues.
    eig_val_D, matrix_p_to_NO_full = np.linalg.eigh(ground_state_dencity_matrix)
    
    # Create a list of indecies associated to eigenstates in decreasing order.
    index_d = np.argsort(-eig_val_D)

    # Makes a change of basis matrix.
    matrix_p_to_NO = matrix_p_to_NO_full[:, index_d[:state]]

    # Reshapes the Potential energy matrix.
    V = V.reshape(state_old**2,state_old**2)

    # Apply the change of pasis to Kinetic energy matrix.
    K_mu = matrix_p_to_NO.T.conj() @ K @ matrix_p_to_NO

    # Create a change of basis matrix for a reshaped potential.
    matrix_p_to_NO_V = np.kron(matrix_p_to_NO, matrix_p_to_NO)

    # Apply the change of pasis to Potential energy matrix.
    V_mu = matrix_p_to_NO_V.T.conj() @ V @ matrix_p_to_NO_V

    # Import the changed Kinetic and Potential energy matrices to create the associated hamiltonial.
    H_mu = hamiltonian_dense(state, site, g_val, tau, l_val, K_mu, V_mu, True)[0]

    # It is importatnt to keep the return in this format since hamiltonian_general uses this structure. 
    return H_mu, K_mu, V_mu

def hamiltonian_general_dense(states: int, sites: int, g_val: float, tau: float=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Interates through systems progressively increasing the nubmer of cites.

    Parameters
    ----------
    state : int
        Total number states in the system, counting the ground state. Ex: system of -1, 0, 1 would be a system of 3 states.
    site : int
        The number of rotors in the system.
    g_val : float
        The constant multiplier for the potential energy. Usually in the range of 0 <= g <= 1.
    tau : float, optional
        Dipolar plains chain angle.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Returns a tuple of Kinetic energy matrix, Potential energy matrix and a Hamiltonian in the respective order.
        Dence Kinetic energy matrix in basis p and dimension of (state, state).
        Dence Potential energy matrix n basis p and shape of (state, state, state, state) simetric along the diagonal.
        Dence Hamiltonian of shape (state^site, state^site) constructed from above Kinetic and Potential.
    """
    # Create the original full hamiltonian
    H_K_V = hamiltonian_dense(11, 3, g_val, tau)
    
    # Iterate through every new system by increasing the size of the system by 2 sites every eteration.
    for current_site in range(3, sites + 2, 2):
        # Make an approximation of the Hamiltonian, Kinetic and Potential energy matrices for the bigger system.
        H_K_V = hamiltonian_big_dense(states, current_site, g_val, H_K_V, tau)

    return H_K_V[0], H_K_V[1], H_K_V[2]