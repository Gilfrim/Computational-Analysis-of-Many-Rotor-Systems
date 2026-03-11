import numpy as np
from scipy.sparse import diags

from quant_rotor.Hamiltonian_models.Dense.support_ham import (
    basis_m_to_p_matrix_conversion,
    write_matrix_elements,
)


def rotor_operators(
    state: int,
    g_val: float,
    psi_twist: float = 0,
    lambda_val: float = 1,
    D: float = 1,
    field: bool = False,
):

    # Create a Kinetic and Potential energy matricies.
    K, V = write_matrix_elements((state - 1) // 2, psi_twist)

    # Optional modifier for the one body (Kinetic energy) operator.
    # Creates and adds a tridiagonal matrix with 0 along the center diagonal and two shifted diagonals determened
    # by the value "l" to the Kinetic energy matrix.

    # Reshape a potential energy matrix from (state^2, state^2) -> (state, state, state, state).
    V_tensor = V.reshape(state, state, state, state)

    # Transform Kinnetic and Potential energy matricies from m basis to p basis.
    K_in_p = basis_m_to_p_matrix_conversion(K, state)
    V_in_p = basis_m_to_p_matrix_conversion(V_tensor, state)

    # Reshape a Potential energy matrix back from (state, state, state, state) -> (state^2, state^2).
    K_in_p = K_in_p * D
    V_in_p = V_in_p.reshape(state**2, state**2) * g_val

    if field:
        l = (1 / np.sqrt(2)) * lambda_val
        L_sparce = diags([l, 0, l], offsets=[-1, 0, 1], shape=(state, state))
        L_dense = L_sparce.toarray()

        K_in_p = K_in_p + L_dense

    return K_in_p, V_in_p


def heisenberg_operators(
    J_x: int,
    J_y: int,
    J_z: int,
    h_x: int,
    h_y: int,
    h_z: int,
    g_val: float,
    lambda_val: int = 1,
    D: float = 1,
    field: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    state = 2
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    sigma_x_2 = np.kron(sigma_x, sigma_x)
    sigma_y_2 = np.kron(sigma_y, sigma_y)
    sigma_z_2 = np.kron(sigma_z, sigma_z)

    K = (h_x * sigma_x + h_y * sigma_y + h_z * sigma_z) * D
    V = (J_x * sigma_x_2 + J_y * sigma_y_2 + J_z * sigma_z_2).reshape(
        state**2, state**2
    ) * g_val

    if field:
        l = (1 / np.sqrt(2)) * lambda_val
        L_sparce = diags([l, 0, l], offsets=[-1, 0, 1], shape=(state, state))
        L_dense = L_sparce.toarray()

        K = K + L_dense

    return K, V
