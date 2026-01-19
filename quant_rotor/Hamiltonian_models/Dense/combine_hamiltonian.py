import numpy as np

from quant_rotor.Hamiltonian_models.Dense.rotor_hamiltonian import hamiltonian_dense
from quant_rotor.Hamiltonian_models.Dense.support_ham import (
    H_kinetic,
    H_potential_combined,
    V_double,
)


def combine_hamiltonian(
    site_original: int,
    site_combined: int,
    K: np.ndarray,
    V: np.ndarray,
    periodic: bool,
):

    state_original = K.shape[0]
    state_combined = state_original ** (site_original // site_combined)

    n_sites_combined = site_original // site_combined

    K_combined, _, _ = hamiltonian_dense(
        state_original,
        n_sites_combined,
        1,
        periodic=False,
        Import_K_V=True,
        K_import=K,
        V_import=V,
    )

    V_combined_non_periodic = V_double(
        state_original,
        n_sites_combined,
        V,
        False,
    )

    V_combined_non_periodic = V_double(state_original, n_sites_combined * 2, V, True)

    K_H = H_kinetic(state_combined, site_combined, K_combined)
    V_H = H_potential_combined(
        state_combined,
        site_combined,
        V_combined_non_periodic,
        periodic,
        V_combined_non_periodic,
        True,
    )

    H = K_H + V_H

    return H, K_combined, V_combined_non_periodic, V_combined_non_periodic
