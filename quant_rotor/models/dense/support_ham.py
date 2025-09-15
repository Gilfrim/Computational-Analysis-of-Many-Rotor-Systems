import numpy as np
import scipy.sparse as sp

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

def basis_m_to_p_matrix_conversion(matrix: np.ndarray, state: int)->np.ndarray:
    """
    Permute axes of a vector/tensor from m-basis ordering to p-basis ordering.

    This function applies the p→m inverse index map (built from the basis size)
    across all axes of `matrix` to produce the same data laid out in the
    p-basis order. It works for 1D vectors, 2D matrices, or higher-order tensors
    whose each axis has length ``d = state = 2*n + 1``.

    The index map is constructed as:
        ``perm = create_inverse_index_map((state - 1) // 2)``
    and then applied to every axis via ``np.ix_``.

    Examples
    --------
    Vector (1D):
    >>> d = 5  # state
    >>> v_m = np.array([-2, -1, 0, 1, 2])   # values aligned with m-order
    >>> v_p = basis_m_to_p_matrix_conversion(v_m, state=d)
    >>> v_p
    array([ 0, -1,  1, -2,  2])

    Matrix (2D): applies the same permutation to both axes.
    For higher-order tensors, the permutation is applied to each axis.

    Parameters
    ----------
    matrix : np.ndarray
        Input array in m-basis ordering. Can be 1D, 2D, or ND, but each axis
        is expected to have length ``state``.
    state : int
        Total basis size ``d = 2*n + 1``. (Here `n` is the number of unique
        positive/negative momentum states.)

    Returns
    -------
    np.ndarray
        A view/copy of `matrix` with all axes permuted into p-basis order.

    Notes
    -----
    - This routine assumes **all axes** correspond to the same basis and
      therefore uses the **same permutation** for each axis.
    - If only some axes represent basis indices, apply the permutation
      selectively to those axes instead of using `np.ix_` on all.
    """

    # Number of dimensions of the input tensor (1D, 2D, ND…)
    dim = matrix.ndim

    # Compute the p→m index map for a basis of size `state`
    # Note: (state - 1) // 2 = numer_unique_states
    index_map = create_inverse_index_map((state - 1) // 2)

    # Replicate the same index map for each axis of the tensor
    index_maps = [index_map] * dim

    # Reorder the tensor along all axes simultaneously using np.ix_
    matrix = matrix[np.ix_(*index_maps)]

    return matrix
 
def write_matrix_elements(numer_unique_states: int, tau: float=0) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct kinetic and potential energy operator matrices for a truncated rotor basis.

    Parameters
    ----------
    numer_unique_states : int
        Number of unique momentum states to include in the truncated basis.
        The total dimension of the basis is given by:
        d = 2 * numer_unique_states + 1.
    tau : float
        Dipolar plains chain angle.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - K: diagonal kinetic energy operator as a dense CSR matrix in the 'p' basis
        - V: dense potential energy operator as a CSR matrix in the 'm' basis
    """

    d = 2 * numer_unique_states + 1

    # Generate Kinetic Energy Matrix
    K = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(i, d):
            K[i, j] = free_one_body(i, j, numer_unique_states)

    # Generate Potential Energy Matrix
    V = np.zeros((d**2, d**2), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    # if k * d + l >= i * d + j:
                        V[i*d + j, k*d + l] = interaction_two_body_coplanar(i, j, k, l, tau)

    return K, V

def H_kinetic(states: int, sites: int, K: np.ndarray) -> np.ndarray:
    """
    Constructs the dense kinetic energy Hamiltonian in a many-body tensor-product space,
    using a single-particle operator K applied independently to each site.

    Parameters
    ----------
    state : int
        Total number states in the system, counting the ground state. Ex: system of -1, 0, 1 would be a system of 3 states.
    site : int
        The number of rotors (sites) in the system.
    K : np.ndarray
        Dense Kinetic energy matrix in the p-basis, shape (state, state).

    Returns
    -------
    np.ndarray
        Dense many-body kinetic Hamiltonian.
    """

    # Create a matrix of the shape of Kinetic energy Hamiltonian filled with zeros.
    K_H = np.zeros((states**sites, states**sites), dtype=complex)

    for x in range(sites):

        # Define the total number of elements in the matrix operator, which represent the left and right sites that are not interacting 
        # by n_lambda and n_mu respectively.
        n_lambda = states**(x)
        n_mu = states**(sites - x - 1)

        # Iterate through all elements of the Kinetic energy matrix operator.
        for p in range(states):
            for p_prime in range(states):

                # Extract an associated element.
                val = K[p, p_prime]

                # Check if element is non zero.
                if val == 0:
                    continue  # skip writing 0s

                for Lambda in range(int(n_lambda)):
                    for mu in range(int(n_mu)):

                        # Calculate the indices in the hamiltonian.
                        i = mu + p * n_mu + Lambda * states * n_mu
                        j = mu + p_prime * n_mu + Lambda * states * n_mu

                        # Assign a values to associated.
                        K_H[i, j] += val
    return K_H
 
def H_potential(states: int, sites: int, V: np.ndarray, g_val: float, periodic: bool) -> np.ndarray:
    """
    Constructs the dense Potential energy Hamiltonian using a two-site interaction operator V
    and coupling constant g_val. The interaction acts on nearest-neighbor pairs (periodic).

    Parameters
    ----------
    state : int
        Total number states in the system, counting the ground state. Ex: system of -1, 0, 1 would be a system of 3 states.
    site : int
        The number of rotors (sites) in the system.
    V : np.ndarray
        Dense potential energy matrix in the p-basis, shape (state² * state²).
    g_val : float
        The constant multiplier for the Potential energy. Typically in the range 0 <= g <= 1.
    periodic : bool
        Defines if the hamiltonian for the periodic system or the non-peirodic system.

    Returns
    -------
    np.ndarray
        Sparse many-body Potential energy Hamiltonian operator.
    """

    # Create a matrix of the shape of Potential energy Hamiltonian filled with zeros.
    V_H = np.zeros((states**sites, states**sites), dtype=complex)

    if periodic:
        site_range = sites
    else:
        site_range = sites-1

    for x in range(site_range):
        # With x defining the first site of two-body interaction, we define the second dynamically.
        y = (x+1) % sites

        # Define the total number of elements in the matrix operator, which represent the left, right, and center sites that are not interacting 
        # by n_lambda, n_mu, n_nu, respectively.
        n_lambda = states**(x % (sites-1))
        n_mu = states**((sites - y - 1) % (sites - 1))
        n_nu = states**(np.abs(y - x) - 1)

        # Iterate through all elements of the Potential energy matrix operator.
        for q in range(states):
            for q_prime in range(states):
                for p in range(states):
                    for p_prime in range(states):

                        # Calculate the flattened indices of the associated element.
                        row = p * states + q
                        col = p_prime * states + q_prime
                        val = V[row, col]

                        # Check if element is non zero.
                        if val == 0:
                            continue  # skip writing 0s

                        for Lambda in range(int(n_lambda)):
                            for mu in range(int(n_mu)):
                                for nu in range(int(n_nu)):
                                    
                                    # Calculate the indices in the hamiltonian.
                                    i = mu + q*n_mu + nu*states*n_mu + p*n_nu*n_mu*states + Lambda*n_nu*n_mu*states**2
                                    j = mu + q_prime*n_mu + nu*states*n_mu + p_prime*n_nu*n_mu*states + Lambda*n_nu*n_mu*states**2
                                    
                                    # Assign a values to associated. 
                                    V_H[i, j] += val * g_val
    return V_H
 
def free_one_body(i: int, j: int, max_m: int) -> float:
    """
    Computes the matrix element ⟨i|K|j⟩ of the free particle Hamiltonian on a ring
    in the momentum representation.

    This represents the kinetic energy operator for a quantum rotor (free particle on a ring),
    which is diagonal in the momentum basis:
        K = ∑ₘ (m - max_m)² |m⟩⟨m|

    Parameters
    ----------
    i : int
        Row index in the basis (momentum state index).
    j : int
        Column index in the basis (momentum state index).
    max_m : int
        The shift for centering the angular momentum indices; usually max_m = floor((d - 1)/2)
        for `d` total states. This centers the momentum spectrum around zero.

    Returns
    -------
    float
        The value of the kinetic energy matrix element ⟨i|K|j⟩. Nonzero only when i == j.
    """
    if i == j:
        return (i - max_m) ** 2
    else:
        return 0.0
 
def interaction_two_body_coplanar(i1: int, i2: int, j1: int, j2: int, tau: float) -> complex:
    """
    Computes the two-body matrix element ⟨i1, i2|V|j1, j2⟩ of the dipole-dipole interaction
    between two planar rotors aligned along the x-axis, in the momentum basis.

    This implementation models the coplanar dipole-dipole interaction using the momentum-shift
    selection rules described in:
        https://arxiv.org/abs/2401.02887

    The interaction Hamiltonian is of the form:
        V = 0.75 ∑ₘ₁,ₘ₂ (|m₁, m₂⟩⟨m₁±1, m₂±1|) 
            - 0.25 ∑ₘ₁,ₘ₂ (|m₁, m₂⟩⟨m₁±1, m₂∓1|)

    Only matrix elements where both indices differ by ±1 are non-zero.

    Parameters
    ----------
    i1 : int
        First particle's row index (bra momentum).
    i2 : int
        Second particle's row index (bra momentum).
    j1 : int
        First particle's column index (ket momentum).
    j2 : int
        Second particle's column index (ket momentum).
    tau : float
        Dipolar plains chain angle.

    Returns
    -------
    complex
        The matrix element ⟨i1, i2|V|j1, j2⟩. Non-zero only if |i1 - j1| = |i2 - j2| = 1.
    """
    # Selection rule: both i1 and i2 must differ by ±1 from j1 and j2 respectively.
    if (abs(i1 - j1) != 1) or (abs(i2 - j2) != 1):
        return 0.0

    # Coefficients based on the specific momentum exchange:
    # (++ or --) →  +0.75
    # (+-, -+)   →  -0.25

    if i1 == j1 + 1:
        if i2 == j2 + 1:
            # print(f"{i1}, {j1}, {i2}, {j2} --> 0.75")
            return 0.75 * np.exp(1j * 2 * tau)  # ⟨m1+1, m2+1|
        else:
            # print(f"{i1}, {j1}, {i2}, {j2} --> -0.25")
            return -0.25 # ⟨m1+1, m2−1|
    else:
        if i2 == j2 + 1:
            # print(f"{i1}, {j1}, {i2}, {j2} --> -0.25")
            return -0.25 # ⟨m1−1, m2+1|
        else:
            # print(f"{i1}, {j1}, {i2}, {j2} --> 0.75")
            return 0.75 * np.exp(1j * 2 * tau)  # ⟨m1−1, m2−1|