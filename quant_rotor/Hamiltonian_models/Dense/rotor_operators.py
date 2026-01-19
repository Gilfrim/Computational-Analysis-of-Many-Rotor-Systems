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
