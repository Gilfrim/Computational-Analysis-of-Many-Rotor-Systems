import numpy as np
import opt_einsum as oe
import scipy.sparse as sp

from quant_rotor.Hamiltonian_models.Dense.density_matrix import density_matrix_1
from quant_rotor.Hamiltonian_models.Dense.hamiltonian import hamiltonian_dense


def NO_transform(
    NO_number: int,
    state_orig: int,
    H: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    periodic : bool
        Defines if the hamiltonian for the periodic system or the non-peirodic system.
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

    # Read the site and state of the original system.
    site_orig = int(np.log(H.shape[0]) / np.log(state_orig))

    # Extract eigenstate and eigenvectors from the original system hamiltonian.
    eig_val, eig_vec = np.linalg.eigh(H)

    # Find the index associated with the smallest eigenstate.

    index = np.argmin(eig_val)
    ground_state_vec = eig_vec[:, index]

    # Make a one site dencity matrix associated with the ground state.
    ground_state_dencity_matrix = density_matrix_1(
        state_orig, site_orig, ground_state_vec, 0
    )

    # Extract eigenstates and eigenvalues.
    eig_val_D, matrix_p_to_NO_full = np.linalg.eigh(ground_state_dencity_matrix)

    # Create a list of indecies associated to eigenstates in decreasing order.
    index_d = np.argsort(-eig_val_D)

    print(eig_val_D)
    print(matrix_p_to_NO_full)

    # Makes a change of basis matrix.
    matrix_p_to_NO = matrix_p_to_NO_full[:, index_d[:NO_number]]

    # Apply the change of basis to Kinetic energy matrix.
    K_NO = matrix_p_to_NO.T.conj() @ K @ matrix_p_to_NO

    # Create a change of basis matrix for a reshaped potential.
    matrix_p_to_NO_V = np.kron(matrix_p_to_NO, matrix_p_to_NO)

    # Apply the change of basis to Potential energy matrix.
    V_NO = matrix_p_to_NO_V.conj().T @ V @ matrix_p_to_NO_V

    # It is importatnt to keep the return in this format since hamiltonian_general uses this structure.
    return K_NO, V_NO, matrix_p_to_NO_full
