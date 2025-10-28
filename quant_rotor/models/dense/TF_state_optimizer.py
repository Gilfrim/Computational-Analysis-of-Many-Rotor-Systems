import matplotlib.pyplot as plt
import numpy as np
import opt_einsum as oe
from scipy.optimize import minimize

from quant_rotor.core.dense.hamiltonian import hamiltonian_dense
from quant_rotor.core.dense.t_amplitudes_guess import amplitute_energy
from quant_rotor.models.dense.density_matrix import density_matrix_1


def TF_optimizer(
    site: int, state: int, g: float, D: float, K: np.ndarray, V: np.ndarray
) -> np.ndarray:

    H, K, V = hamiltonian_dense(state, site, g)
