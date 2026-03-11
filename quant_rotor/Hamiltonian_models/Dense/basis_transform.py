import numpy as np

from quant_rotor.CCC_integration_methods.Dense.thermofield_boltz_funcs import (
    thermofield_change_of_basis,
)
from quant_rotor.Hamiltonian_models.Dense.density_matrix import density_matrix_1
from quant_rotor.Hamiltonian_models.Dense.hamiltonian import hamiltonian_dense
from quant_rotor.Hamiltonian_models.Dense.support_ham import V_double_xy


def energy_transform(
    K: np.ndarray, V_xy: np.ndarray, V_yx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, energy_basis = np.linalg.eigh(K)

    # Apply the change of basis to Kinetic energy matrix.
    K_e = energy_basis.T.conj() @ K @ energy_basis

    # Create a change of basis matrix for a reshaped potential.
    energy_basis_V = np.kron(energy_basis, energy_basis)

    # Apply the change of basis to Potential energy matrix.
    V_e_xy = energy_basis_V.conj().T @ V_xy @ energy_basis_V
    V_e_yx = energy_basis_V.conj().T @ V_yx @ energy_basis_V

    return K_e, V_e_xy, V_e_yx


def TF_transform(
    K: np.ndarray, V_xy: np.ndarray, V_yx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    state = K.shape[0]
    state_TF = state**2

    U, _ = thermofield_change_of_basis(K)

    I = np.eye(state)

    K_prim = np.einsum("pq,mw->pmqw", K, I, optimize="optimal").reshape(
        state_TF, state_TF
    )

    K_tilda = U.T @ K_prim @ U

    V_tensor_xy = V_xy.reshape(state, state, state, state)
    V_tensor_yx = V_yx.reshape(state, state, state, state)

    V_prim_xy = np.einsum(
        "pqrs,mw,nv->pmqnrwsv", V_tensor_xy, I, I, optimize="optimal"
    ).reshape(state_TF**2, state_TF**2)

    V_prim_yx = np.einsum(
        "pqrs,mw,nv->pmqnrwsv", V_tensor_yx, I, I, optimize="optimal"
    ).reshape(state_TF**2, state_TF**2)

    V_grouped_xy = V_prim_xy.reshape(state_TF, state_TF, state_TF, state_TF)
    V_grouped_yx = V_prim_yx.reshape(state_TF, state_TF, state_TF, state_TF)

    V_tilda_xy = np.einsum(
        "Mi,Wj,ijab,aN,bV->MWNV", U.T, U.T, V_grouped_xy, U, U, optimize="optimal"
    ).reshape(state_TF**2, state_TF**2)

    V_tilda_yx = np.einsum(
        "Mi,Wj,ijab,aN,bV->MWNV", U.T, U.T, V_grouped_yx, U, U, optimize="optimal"
    ).reshape(state_TF**2, state_TF**2)

    return K_tilda, V_tilda_xy, V_tilda_yx


def combine_transform(
    n_sites_combined: int,
    K: np.ndarray,
    V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    state_original = K.shape[0]

    K_combined = hamiltonian_dense(n_sites_combined, K, V, V, False)

    V_combined_xy = V_double_xy(
        state_original,
        n_sites_combined * 2,
        V,
        False,
    )

    V_combined_yx = V_double_xy(
        state_original,
        n_sites_combined * 2,
        V,
        True,
    )

    return K_combined, V_combined_xy, V_combined_yx


def NO_transform(
    NO_number: int,
    H: np.ndarray,
    K: np.ndarray,
    V_xy: np.ndarray,
    V_yx: np.ndarray,
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
    state_orig = K.shape[0]
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

    # Makes a change of basis matrix.
    matrix_p_to_NO = matrix_p_to_NO_full[:, index_d[:NO_number]]

    # Apply the change of basis to Kinetic energy matrix.
    K_NO = matrix_p_to_NO.T.conj() @ K @ matrix_p_to_NO

    # Create a change of basis matrix for a reshaped potential.
    matrix_p_to_NO_V = np.kron(matrix_p_to_NO, matrix_p_to_NO)

    # Apply the change of basis to Potential energy matrix.
    V_NO_xy = matrix_p_to_NO_V.conj().T @ V_xy @ matrix_p_to_NO_V
    V_NO_yx = matrix_p_to_NO_V.conj().T @ V_yx @ matrix_p_to_NO_V

    # It is importatnt to keep the return in this format since hamiltonian_general uses this structure.
    return K_NO, V_NO_xy, V_NO_yx, matrix_p_to_NO_full
