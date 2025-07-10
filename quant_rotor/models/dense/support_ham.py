import numpy as np

def m_to_p(energy_state:int)->int:
    """
    Taking am energy state returns an indes in p basis that this state is associated with. 

    Expample is given with a vector which would be done by calling this function in a for loop inerating over the elements of the vector
    or by vectorysing the function.

    Ex: (-2, -1, 0, 1, 2) -> (3, 1, 0, 2, 4)

    Logic: Sicne in the p basis the vector would look like this:(0, -1, 1, -2, 2) (in our case the it doesn't matter if the positive
    is first or nevative since the values are getting squared later) 
    To get a mappint from (-2, -1, 0, 1, 2) -> (0, -1, 1, -2, 2) we construct a piecevise function.

    f(x) = {2|x+1| + 1, x < 0; 2x, x >= 0} 

    Parameters
    ----------
    energy_state : int
        An energy state positive or negative.

    Returns
    -------
    int
        Returns an index associated with a given energy state.
    """
    return 2 * abs(energy_state + 1) + 1 if energy_state < 0 else 2 * energy_state

def create_inverse_index_map(numer_unique_states: int) -> np.ndarray:
    """
    Creates an one dimentional (index map for a vector in m -> p) inverce index map of the permutation m -> p by:
    1. Creating a vector in m.
    2. Creating an index map m -> p.
    3. Creating a reverce index map which would serve as an imput into a np.ix_ function.

    Example: 
    
    vec_in_m = (-2, -1, 0, 1, 2)
    m -> p : (3, 1, 0, 2, 4)
    p -> m : (2, 1, 3, 0, 4)

    vec_in_p = (0, -1, 1, -2, 2) = (-2, -1, 0, 1, 2)[np.ix_(2, 1, 3, 0, 4)]

    What np.ix_ does is it takes any index map p -> m and preforms an operation an index of the value in index map into a 
    plase of the index. So if we take vec_in_m: vec_in_p[0] = vec_in_m[2], vec_in_p[1] = vec_in_m[1], vec_in_p[2] = vec_in_m[3], etc.

    Parameters
    ----------
    numer_unique_states : int
        Number of unique states in the system, not counting the ground state. The number of unique states can be calculated 
        acording to this formulaL f(x) = (x-1)/2
        Example: system of -1, 0, 1 → system of 1 site = 3 states.

    Returns
    -------
    np.ndarray
        Returns an inverce map vector. 
    """
    # Create a vercor in m basis with the aproriate number of states. Ex.(state = 5): vector_in_m = (-2, -1, 0, 1, 2)
    vector_in_m = np.arange(-numer_unique_states, numer_unique_states + 1)

    # Create an index map vector m -> p basis.
    index_map_m_to_p = np.vectorize(m_to_p)(vector_in_m)

    # Create inverse index map vector p -> m
    inverse_index_map = np.zeros_like(index_map_m_to_p)
    for i, p in enumerate(index_map_m_to_p):
        inverse_index_map[p] = i

    return inverse_index_map

def basis_m_to_p_matrix_conversion(matrix: np.ndarray)->np.ndarray:
    """_summary_

    Parameters
    ----------
    matrix : np.ndarray
        _description_

    Returns
    -------
    int
        _description_
    """

    dim = matrix.ndim
    index_map = create_inverse_index_map((matrix.shape[0]-1)//2)

    index_maps = []
    for i in range(dim):
        index_maps.append(index_map)

    matrix = matrix[np.ix_(*index_maps)]

    return matrix
 
def write_matrix_elements(numer_unique_states):
    d = 2 * numer_unique_states + 1

    # Generate Kinetic Energy Matrix
    K = np.diag(np.arange(-numer_unique_states, numer_unique_states + 1))

    # Generate Potential Energy Matrix
    V = np.zeros((d**2, d**2))
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    if k * d + l >= i * d + j:
                        V[i*d + j, k*d + l] = interaction_two_body_coplanar(i, j, k, l)

    return K, V

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