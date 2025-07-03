import numpy as np
from quant_rotor.core.dense import hamiltonian

def test_dense_hamiltonian_shape():
    H = hamiltonian.hamiltonian(3, 3, 1)
    assert isinstance(H, np.ndarray)
    assert H.shape == (27, 27)