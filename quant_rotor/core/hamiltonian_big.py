import numpy as np
from scipy.sparse import csr_matrix, kron
from scipy.sparse.linalg import eigsh
from quant_rotor.core.hamiltonian import hamiltonian
from quant_rotor.models.density_matrix import density_matrix_1

def hamiltonian_big(state: int, site: int, g_val: float, H_K_V: list[np.ndarray], l_val: float=0) -> np.ndarray:

    H = H_K_V[0]
    K = H_K_V[1]
    V = H_K_V[2]

    state_old = int(K.shape[0])
    site_old = int(np.log(H.shape[0]) / np.log(state_old))

    H_sparse_csr = H.tocsr()

    eig_val, eig_vec = eigsh(H_sparse_csr, k=1, which='SA')

    # index = np.argsort(eig_val)
    # psi_vec = eig_vec[:, index[0]] 
    rho_site_0 = density_matrix_1(state_old, site_old, eig_vec, 0)

    eig_val_D, matrix_p_to_natural_orbital = np.linalg.eigh(rho_site_0)

    # eig_val_D, eig_vec_D = np.linalg.eig(rho_site_0)
    index_d = np.argsort(-eig_val_D)

    print("Done change of basis.")

    matrix_p_to_natural_orbital_sparse = csr_matrix(matrix_p_to_natural_orbital[:, index_d[:state]])

    # V = V.reshape(state_old**2,state_old**2)
    K_mu = matrix_p_to_natural_orbital_sparse.T.conj() @ K @ matrix_p_to_natural_orbital_sparse
    # V_mu = np.kron(matrix_p_to_natural_orbital_sparse.T.conj(), matrix_p_to_natural_orbital_sparse.T.conj()) @ V @ np.kron(matrix_p_to_natural_orbital_sparse, matrix_p_to_natural_orbital_sparse)

    left = kron(matrix_p_to_natural_orbital_sparse.T.conj(), matrix_p_to_natural_orbital_sparse.T.conj())
    right = kron(matrix_p_to_natural_orbital_sparse, matrix_p_to_natural_orbital_sparse)

    V_mu = left @ V @ right

    # V_mu = V_mu.reshape(state,state,state,state)

    H_mu = hamiltonian(state, site, g_val, l_val, K_mu, V_mu, True, True)[0]

    return H_mu, K_mu, V_mu, matrix_p_to_natural_orbital[:, index_d[:state]]

def hamiltonian_general(states: int, sites: int, g_val: float):

    H_K_V = hamiltonian(11, 3, g_val, spar=True)
    for current_site in range(3, sites + 2, 2):
        H_K_V = hamiltonian_big(states, current_site, g_val, H_K_V)
        print(f"Done {states} states {sites} sites:")

    return H_K_V