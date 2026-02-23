import numpy as np
from scipy.sparse import diags

from quant_rotor.Hamiltonian_models.Dense.support_ham import (
    H_kinetic,
    H_potential_combined,
)


def hamiltonian_dense(
    site: int,
    J_x: int,
    J_y: int,
    J_z: int,
    h_x: int,
    h_y: int,
    h_z: int,
    g_val: float,
    lambda_val: int = 1,
    D: float = 1,
    periodic: bool = True,
    field: bool = False,
    Import: bool = False,
    sigma_x_i: np.ndarray = [],
    sigma_y_i: np.ndarray = [],
    sigma_z_i: np.ndarray = [],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    state = 2
    if Import:
        sigma_x = sigma_x_i
        sigma_y = sigma_y_i
        sigma_z = sigma_z_i
    else:
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    sigma_x_2 = np.kron(sigma_x, sigma_x)
    sigma_y_2 = np.kron(sigma_y, sigma_y)
    sigma_z_2 = np.kron(sigma_z, sigma_z)

    K = h_x * sigma_x + h_y * sigma_y + h_z * sigma_z
    V = (J_x * sigma_x_2 + J_y * sigma_y_2 + J_z * sigma_z_2) * g_val

    O_1_x = H_kinetic(2, site, sigma_x)
    O_1_y = H_kinetic(2, site, sigma_y)
    O_1_z = H_kinetic(2, site, sigma_z)

    O_2_x = H_potential_combined(2, site, sigma_x_2, periodic)
    O_2_y = H_potential_combined(2, site, sigma_y_2, periodic)
    O_2_z = H_potential_combined(2, site, sigma_z_2, periodic)

    V_2 = H_potential_combined(2, site, V, periodic)

    # Reshape a Potential energy matrix back from (state, state, state, state) -> (state^2, state^2).
    O_1 = (h_x * O_1_x + h_y * O_1_y + h_z * O_1_z) * D
    O_2 = (J_x * O_2_x + J_y * O_2_y + J_z * O_2_z) * g_val

    print(np.allclose(O_2, V_2))

    if field:
        l = (1 / np.sqrt(2)) * lambda_val
        L_sparce = diags([l, 0, l], offsets=[-1, 0, 1], shape=(state, state))
        L_dense = L_sparce.toarray()

        O_1 = O_1 + L_dense

    # Add to get the final hamiltonian.
    H_final = O_1 + O_2

    return H_final, K, V
