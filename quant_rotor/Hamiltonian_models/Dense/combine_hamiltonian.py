import numpy as np

from quant_rotor.Hamiltonian_models.Dense.hamiltonian import hamiltonian_dense
from quant_rotor.Hamiltonian_models.Dense.support_ham import (
    H_kinetic,
    H_potential_combined,
    V_double_xy,
    V_double_yx,
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

    K_combined = hamiltonian_dense(n_sites_combined, K, V, V, False)

    V_combined_non_periodic_xy = V_double_xy(
        state_original,
        n_sites_combined * 2,
        V,
        False,
    )

    V_combined_non_periodic_yx = V_double_xy(
        state_original,
        n_sites_combined * 2,
        V,
        True,
    )

    # V_total = V_combined_non_periodic_xy + V_combined_non_periodic_yx

    # print(np.max(np.abs(V_total - V_total.T)))

    K_H = H_kinetic(state_combined, site_combined, K_combined)
    V_H = H_potential_combined(
        state_combined,
        site_combined,
        V_combined_non_periodic_xy,
        periodic,
        V_combined_non_periodic_yx,
    )

    H = K_H + V_H

    return H, K_combined, V_combined_non_periodic_xy, V_combined_non_periodic_yx
