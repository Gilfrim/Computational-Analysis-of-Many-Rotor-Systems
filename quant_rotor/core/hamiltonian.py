import numpy as np
import quant_rotor.models.functions as func
import os
import quant_rotor.models.hamiltonianGenerator as hg
from importlib.resources import files

np.set_printoptions(suppress = True, linewidth = 1500, threshold = 10000, precision = 9)


def create_inverse_index_map(total_num_states: int) -> np.ndarray:
    # map m → p
    m_vals = np.arange(-total_num_states, total_num_states + 1)
    p_vals = np.vectorize(func.m_to_p)(m_vals)

    # Create inverse: map p → position in m-sorted order
    inverse_index_map = np.zeros_like(p_vals)
    for i, p in enumerate(p_vals):
        inverse_index_map[p] = i

    return inverse_index_map

def basis_m_to_p_matrix_conversion(V: np.ndarray)->np.ndarray:

    dim = V.ndim
    index_map = create_inverse_index_map((V.shape[0]-1)//2)

    index_maps = []
    for i in range(dim):
        index_maps.append(index_map)

    V = V[np.ix_(*index_maps)]

    return V

def H_kinetic(states: int, sites: int, h_pp: np.ndarray) -> np.ndarray:

    K = np.zeros((states**sites, states**sites), dtype=complex)

    for x in range(sites):
        n_lambda = states**(x)
        n_mu = states**(sites - x - 1)

        for p in range(states):
            for p_prime in range(states):
                val = h_pp[p, p_prime]
                if val == 0:
                    continue  # skip writing 0s
                for Lambda in range(int(n_lambda)):
                    for mu in range(int(n_mu)):
                        i = mu + p * n_mu + Lambda * states * n_mu
                        j = mu + p_prime * n_mu + Lambda * states * n_mu
                        K[i, j] += val
    return K

def H_potential(states: int, sites: int, h_pp_qq: np.ndarray, g_val: float) -> np.ndarray:

    V = np.zeros((states**sites, states**sites), dtype=complex)

    for x in range(sites):
        y = (x+1) % sites
        n_lambda = states**(x % (sites-1))
        n_mu = states**((sites - y - 1) % (sites - 1))
        n_nu = states**(np.abs(y - x) - 1)

        for q in range(states):
            for q_prime in range(states):
                for p in range(states):
                    for p_prime in range(states):

                        val = h_pp_qq[p,q, p_prime, q_prime]
                        if val == 0:
                            continue  # skip writing 0s

                        for Lambda in range(int(n_lambda)):
                            for mu in range(int(n_mu)):
                                for nu in range(int(n_nu)):
                                    
                                    i = mu + q*n_mu + nu*states*n_mu + p*n_nu*n_mu*states + Lambda*n_nu*n_mu*states**2
                                    j = mu + q_prime*n_mu + nu*states*n_mu + p_prime*n_nu*n_mu*states + Lambda*n_nu*n_mu*states**2
                                    V[i, j] += val * g_val
    return V

def write_matrix_elements(m_max):
    d = 2 * m_max + 1

    # Generate Kinetic Energy Matrix
    K = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            K[i, j] = hg.free_one_body(i, j, m_max)

    # Generate Potential Energy Matrix
    V = np.zeros((d**2, d**2))
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    if k * d + l >= i * d + j:
                        V[i*d + j, k*d + l] = hg.interaction_two_body_coplanar(i, j, k, l)

    # # Write to package's data directory
    # data_dir = files("quant_rotor.data")
    # np.save(os.path.join(data_dir, "K_matrix.npy"), K)
    # np.save(os.path.join(data_dir, "V_matrix.npy"), V)

    return K, V

def hamiltonian(state: int, site: int, g_val: float=1, K_import: np.ndarray=[], V_import: np.ndarray=[], Import: bool=False, clean: bool=False)->np.ndarray:
    if Import == False:
        K_from_npy, V_from_npy = write_matrix_elements((state-1) // 2)

        # data_dir = files("quant_rotor.data")
        # K_from_npy = np.load(data_dir / "K_matrix.npy")
        # V_from_npy = np.load(data_dir / "V_matrix.npy")

        V_from_npy = V_from_npy + V_from_npy.T - np.diag(np.diag(V_from_npy))
        V_tensor = V_from_npy.reshape(state, state, state, state)  # Adjust if needed

        K_in_p = basis_m_to_p_matrix_conversion(K_from_npy)
        V_in_p = basis_m_to_p_matrix_conversion(V_tensor)

        # if clean:
        #     try:
        #         os.remove(data_dir / "K_matrix.npy")
        #         os.remove(data_dir / "V_matrix.npy")
        #     except FileNotFoundError as e:
        #         print(f"File not found during deletion: {e.filename}")

    else:
        K_in_p = K_import
        V_in_p = V_import
        state = K_import.shape[0]

    K_final = H_kinetic(state, site, K_in_p)
    V_final = H_potential(state, site, V_in_p, g_val)
    H_final = K_final + V_final



    return H_final, K_in_p, V_in_p

if __name__ == "__main__":
    site = 3
    state = 3
    print(hamiltonian(state, site))