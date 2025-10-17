import itertools

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

def vector_in_p(state: int) -> np.ndarray:
    """
    Generates a 1D integer array representing the diagonal of Kinetic energy matrix in the 'p' basis.

    The returned vector alternates between non-positive and positive integer state
    in the following order: (0, -1, 1, -2, 2, -3, 3, ...), up to the specified length.
    This indexing is useful in quantum rotor models and similar systems where
    the natural ordering of angular momentum state is rearranged to group
    symmetric contributions.

    Parameters
    ----------
    state : int
        Total number state in the system, counting the ground state. Ex: system of -1, 0, 1 would be a system of 3 state.

    Returns
    -------
    np.ndarray
        A 1D NumPy array of length `state` containing the Kinetic energy matrix diagonal entry.
    """

    # Generate values 0, 1, 2, ..., state - 1
    k = np.arange(state)

    # Apply alternating transformation: 0, -1, 1, -2, 2, ...
    return ((-1)**k) * ((k + 1) // 2)

def build_V_prime_in_p(state: int, tau: float) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Constructs:
    - K: a diagonal kinetic energy operator in the 'p' basis
    - V: a sparse potential energy operator directly in the 'p' basis

    Avoids constructing in the m basis and transforming later.
    """

    dim_V = state **4

    # Construct a diagonal of Kinetic energy matrix in m basis.
    m_vals = np.arange(-(state - 1) // 2, (state - 1) // 2 + 1)

    # Construct index map m -> p.
    perm = np.vectorize(m_to_p)(m_vals)  # Maps m-index → p-index

    data_V = []
    rows = []
    cols = []

    def index(p1: int, p2: int) -> int:
        return p1 * state**3 + p2 * state

    # Loop over (m1, m2) — this preserves physics
    for m1 in range(state):
        for m2 in range(state):
            p1 = perm[m1]
            p2 = perm[m2]
            i = index(p1, p2)
            for dm1, dm2, coef in [
                (1, 1, 0.75),
                (-1, -1, 0.75),
                (1, -1, -0.25),
                (-1, 1, -0.25),
            ]:

                m1p = m1 + dm1
                m2p = m2 + dm2
                if 0 <= m1p < state and 0 <= m2p < state:
                    p1p = perm[m1p]
                    p2p = perm[m2p]
                    j = index(p1p, p2p)

                    for shift_state in range(0, state):
                        for shift_index in range(0, state):
                            rows.append(i + shift_index + state**2 * shift_state)
                            cols.append(j + shift_index + state**2 * shift_state)
                            data_V.append(coef)

    V = sp.csr_matrix((data_V, (rows, cols)), shape=(dim_V, dim_V), dtype=np.float64)

    # Diagonal kinetic operator in p basis
    p = vector_in_p(state)
    p_prime = np.array([[i]*state for i in p]).flatten()
    K = sp.diags(p_prime**2, offsets=0, format='csr')

    return K, V

def build_V_in_p(state: int, tau: float=0) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Constructs:
    - K: a diagonal kinetic energy operator in the 'p' basis
    - V: a sparse potential energy operator directly in the 'p' basis

    Avoids constructing in the m basis and transforming later.
    """

    dim_V = state * state

    # Construct a diagonal of Kinetic energy matrix in m basis.
    m_vals = np.arange(-(state - 1) // 2, (state - 1) // 2 + 1)

    # Construct index map m -> p.
    perm = np.vectorize(m_to_p)(m_vals)  # Maps m-index → p-index

    data_V = []
    rows = []
    cols = []

    def index(p1: int, p2: int) -> int:
        return p1 * state + p2

    # Loop over (m1, m2) — this preserves physics
    for m1 in range(state):
        for m2 in range(state):
            p1 = perm[m1]
            p2 = perm[m2]
            i = index(p1, p2)

            for dm1, dm2, coef in [
                (1, 1, 0.75),
                (-1, -1, 0.75),
                (1, -1, -0.25),
                (-1, 1, -0.25),
            ]:
                m1p = m1 + dm1
                m2p = m2 + dm2
                if 0 <= m1p < state and 0 <= m2p < state:
                    p1p = perm[m1p]
                    p2p = perm[m2p]
                    j = index(p1p, p2p)
                    rows.append(i)
                    cols.append(j)
                    data_V.append(coef)

    V = sp.csr_matrix((data_V, (rows, cols)), shape=(dim_V, dim_V), dtype=np.float64)

    # Diagonal kinetic operator in p basis
    p = vector_in_p(state)
    K = sp.diags(p**2, offsets=0, format='csr')

    return K, V

def H_kinetic_sparse(state: int, site: int, K: sp.spmatrix) -> sp.csr_matrix:
    """
    Constructs the sparse kinetic energy Hamiltonian in a many-body tensor-product space,
    using a single-particle operator K applied independently to each site.

    Parameters
    ----------
    state : int
        Total number states in the system, counting the ground state. Ex: system of -1, 0, 1 would be a system of 3 states.
    site : int
        The number of rotors (sites) in the system.
    K : spmatrix
        Sparse Kinetic energy matrix in the p-basis, shape (state, state).

    Returns
    -------
    csr_matrix
        Sparse many-body kinetic Hamiltonian.
    """
    dim = state ** site # Total size the Hamiltonian
    data = []
    rows = []
    cols = []

    # Use COO for efficient iteration
    K = K.tocoo()

    for x in range(site):

        # Define the total number of elements in the matrix operator, which represent the left and right sites that are not interacting
        # by n_lambda and n_mu respectively.
        n_lambda = state ** x
        n_mu = state ** (site - x - 1)

        # Define place values for state based numerical system.
        stride_p = n_mu
        stride_lambda = state * n_mu

        # Create a new list of row, column and data indecies.
        for p, p_prime, val in zip(K.row, K.col, K.data):
            for Lambda in range(n_lambda):

                # Define place values for state based numerical system.
                base = Lambda * stride_lambda

                for mu in range(n_mu):

                    # Calculate the indices in the hamiltonian.
                    i = base + p * stride_p + mu
                    j = base + p_prime * stride_p + mu

                    # Uppend them to a new rows, collumns and data arrays.
                    rows.append(i)
                    cols.append(j)
                    data.append(val)

    return sp.csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=complex)

def H_potential_sparse(state: int, site: int, V: sp.spmatrix, g_val: float) -> sp.csr_matrix:
    """
    Constructs the sparse Potential energy Hamiltonian using a two-site interaction operator V
    and coupling constant g_val. The interaction acts on nearest-neighbor pairs (periodic).

    Parameters
    ----------
    state : int
        Total number states in the system, counting the ground state. Ex: system of -1, 0, 1 would be a system of 3 states.
    site : int
        The number of rotors (sites) in the system.
    V : spmatrix
        Sparse Potential energy matrix in the p-basis, shape (state² * state²).
    g_val : float
        The constant multiplier for the Potential energy. Typically in the range 0 <= g <= 1.

    Returns
    -------
    csr_matrix
        Sparse many-body Potential energy Hamiltonian operator.
    """

    dim = state ** site
    data = []
    rows = []
    cols = []

    # Use COO for fast iteration
    V = V.tocoo()

    for x in range(site):
        # With x defining the first site of two-body interaction, we define the second dynamically.
        y = (x + 1) % site

        # Define the total number of elements in the matrix operator, which represent the left, right, and center sites that are not interacting
        # by n_lambda, n_mu, n_nu, respectively.
        n_lambda = state ** (x % (site - 1))
        n_mu = state ** ((site - y - 1) % (site - 1))
        n_nu = state ** (abs(y - x) - 1)

        # Define place values for state based numerical system.
        stride_q = n_mu
        stride_nu = state * n_mu
        stride_p = n_nu * n_mu * state
        stride_lambda = n_nu * n_mu * state ** 2

        # Create a new list of row, column and data indecies.
        for row_2s, col_2s, val in zip(
            V.row, V.col, V.data
        ):  # Iterate only through non-zero elements of the
            # Unflatten the indices.
            p, q = divmod(row_2s, state)
            p_prime, q_prime = divmod(col_2s, state)

            for Lambda in range(n_lambda):
                for mu in range(n_mu):
                    for nu in range(n_nu):
                        # Calculate the indices in the hamiltonian.
                        i = mu + q * stride_q + nu * stride_nu + p * stride_p + Lambda * stride_lambda
                        j = mu + q_prime * stride_q + nu * stride_nu + p_prime * stride_p + Lambda * stride_lambda

                        # Uppend them to a new rows, collumns and data arrays.
                        rows.append(i)
                        cols.append(j)
                        data.append(val * g_val)

    # Return a new sparse array of the potential energy part of the mamiltonian.
    return sp.csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=complex)
