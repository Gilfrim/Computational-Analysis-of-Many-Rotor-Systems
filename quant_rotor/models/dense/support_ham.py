import quant_rotor.models.dense.functions as func
import numpy as np
from importlib.resources import files
from scipy.sparse import lil_matrix, csr_matrix

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

def H_kinetic_sparse(states: int, sites: int, h_pp: np.ndarray) -> csr_matrix:
    dim = states ** sites
    K_sparse = lil_matrix((dim, dim), dtype=complex)

    for x in range(sites):
        n_lambda = states ** x
        n_mu = states ** (sites - x - 1)

        for p in range(states):
            for p_prime in range(states):
                val = h_pp[p, p_prime]
                if val == 0:
                    continue
                for Lambda in range(n_lambda):
                    for mu in range(n_mu):
                        i = mu + p * n_mu + Lambda * states * n_mu
                        j = mu + p_prime * n_mu + Lambda * states * n_mu
                        K_sparse[i, j] += val

    return K_sparse.tocsr()

 
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
                        row = p * states + q
                        col = p_prime * states + q_prime
                        val = h_pp_qq[row, col]
                        if val == 0:
                            continue  # skip writing 0s

                        for Lambda in range(int(n_lambda)):
                            for mu in range(int(n_mu)):
                                for nu in range(int(n_nu)):
                                    
                                    i = mu + q*n_mu + nu*states*n_mu + p*n_nu*n_mu*states + Lambda*n_nu*n_mu*states**2
                                    j = mu + q_prime*n_mu + nu*states*n_mu + p_prime*n_nu*n_mu*states + Lambda*n_nu*n_mu*states**2
                                    V[i, j] += val * g_val
    return V

def H_potential_sparse(states: int, sites: int, h_pp_qq: np.ndarray, g_val: float) -> csr_matrix:
    dim = states ** sites
    V_sparse = lil_matrix((dim, dim), dtype=complex)
    h_pp_qq = h_pp_qq.tocsr()

    for x in range(sites):
        y = (x + 1) % sites
        n_lambda = states ** (x % (sites - 1))
        n_mu = states ** ((sites - y - 1) % (sites - 1))
        n_nu = states ** (abs(y - x) - 1)

        for q in range(states):
            for q_prime in range(states):
                for p in range(states):
                    for p_prime in range(states):
                        row = p * states + q
                        col = p_prime * states + q_prime
                        val = h_pp_qq[row, col]
                        if val == 0:
                            continue

                        for Lambda in range(n_lambda):
                            for mu in range(n_mu):
                                for nu in range(n_nu):
                                    i = (mu + q * n_mu + nu * states * n_mu +
                                         p * n_nu * n_mu * states +
                                         Lambda * n_nu * n_mu * states ** 2)
                                    j = (mu + q_prime * n_mu + nu * states * n_mu +
                                         p_prime * n_nu * n_mu * states +
                                         Lambda * n_nu * n_mu * states ** 2)
                                    V_sparse[i, j] += val * g_val

    return V_sparse.tocsr()

def H_potential_general(states: int, sites: int, h_pp_qq: np.ndarray, g_val: float) -> np.ndarray:

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
                        row = p * states + q
                        col = p_prime * states + q_prime
                        val = h_pp_qq[row, col]
                        if val == 0:
                            continue  # skip writing 0s

                        for Lambda in range(int(n_lambda)):
                            for mu in range(int(n_mu)):
                                for nu in range(int(n_nu)):
                                    
                                    i = mu + q*n_mu + nu*states*n_mu + p*n_nu*n_mu*states + Lambda*n_nu*n_mu*states**2
                                    j = mu + q_prime*n_mu + nu*states*n_mu + p_prime*n_nu*n_mu*states + Lambda*n_nu*n_mu*states**2
                                    V[i, j] += val
    return V

 
def free_one_body(i, j, max_m):

    # Free particle on a ring Hamiltonian in the momentum representation. 
    # K = \sum_{m=0}^{2l} (m-l)^2 |m><m|
    if i == j:
        return (i-max_m)**2
    else: 
        return 0.0
 
def interaction_two_body_coplanar(i1, i2, j1, j2):

    # returns <i1,i2|V|j1,j2> where V is the dipole-dipole interaction between two planar rotors oriented along the x-axis.
    # See https://arxiv.org/abs/2401.02887 for details. 
    # V = 0.75 \sum_{m1, m2} (|m1, m2> <m1+1, m2+1| + |m1, m2> <m1-1, m2-1|) - 0.25 \sum_{m1, m2} (|m1, m2> <m1-1, m2+1| + |m1, m2> <m1+1, m2-1|)
    if ((abs(i1 - j1) != 1) or (abs(i2 - j2) != 1)):
        return 0.0
    elif (i1 == j1 + 1):
        if (i2 == j2 + 1):
            return 0.75
        else:
            return -0.25
    else:
        if (i2 == j2 + 1):
            return -0.25
        else:
            return 0.75

 
def write_matrix_elements(m_max):
    d = 2 * m_max + 1

    # Generate Kinetic Energy Matrix
    K = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            K[i, j] = free_one_body(i, j, m_max)

    # Generate Potential Energy Matrix
    V = np.zeros((d**2, d**2))
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    if k * d + l >= i * d + j:
                        V[i*d + j, k*d + l] = interaction_two_body_coplanar(i, j, k, l)

    # # Write to package's data directory
    # data_dir = files("quant_rotor.data")
    # np.save(os.path.join(data_dir, "K_matrix.npy"), K)
    # np.save(os.path.join(data_dir, "V_matrix.npy"), V)

    return K, V