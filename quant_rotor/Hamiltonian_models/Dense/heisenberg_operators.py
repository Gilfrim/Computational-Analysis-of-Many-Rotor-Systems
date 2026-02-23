import numpy as np
from scipy.sparse import diags

from quant_rotor.Hamiltonian_models.Dense.support_ham import (
    H_kinetic,
    H_potential_combined,
)


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
