import numpy as np

from quant_rotor.Hamiltonian_models.Dense.support_ham import (
    H_kinetic,
    H_potential_combined,
)


def hamiltonian_dense(
    site: int,
    K: np.ndarray,
    V_xy: np.ndarray,
    V_yx: np.ndarray,
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    state = K.shape[0]

    if periodic:
        V_xy = V_xy
        V_yx = V_yx

    # Construct a Kinetic and Potential hamiltonian.
    K_final = H_kinetic(state, site, K)

    V_final = H_potential_combined(state, site, V_xy, periodic, V_yx)

    # Add to get the final hamiltonian.
    H_final = K_final + V_final

    return H_final
