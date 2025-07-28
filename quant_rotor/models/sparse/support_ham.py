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

def basis_m_to_p_matrix_conversion(V: sp.csr_matrix) -> sp.csr_matrix:
    """
    Performs a basis transformation of a two-body operator matrix `V` from the m-basis to the p-basis.

    The function assumes that `V` is a square sparse matrix of shape (d², d²), where d is the 
    number of one-body states in the m-basis. The transformation uses a predefined mapping 
    `m → p` based on the `m_to_p` function. Each row and column index is interpreted as a pair 
    of one-body m-basis indices, which are mapped to corresponding p-basis indices, and then 
    re-flattened.

    Parameters
    ----------
    V : sp.csr_matrix
        A sparse matrix in CSR format representing a two-body operator in the m-basis.
        The shape of the matrix must be (d², d²), where d is an odd integer.

    Returns
    -------
    sp.csr_matrix
        A sparse matrix in CSR format representing the same two-body operator, but in the p-basis.

    Notes
    -----
    - This function assumes that `m_to_p` is a bijective mapping from m-basis indices 
      (e.g. -2, -1, 0, 1, 2) to p-basis indices (e.g. 3, 1, 0, 2, 4).
    - The p-basis is defined such that the transformation preserves all nonzero elements,
      only rearranging them by index.
    """
    # Extract the # of states from the shape of the matrix.
    state = int(np.sqrt(V.shape[0]))

    # Construct a diagonal of Kinetic energy matrix in m basis.
    m_vals = np.arange(-(state - 1) // 2, (state - 1) // 2 + 1)

    # Construct index map m -> p.
    perm = np.vectorize(m_to_p)(m_vals)  # Maps m-index → p-index

    # Make sure V is in a COO format for faster processing.
    V = V.tocoo()

    # Make emply arrays for row and column elements.
    row_mapped = np.empty_like(V.row)
    col_mapped = np.empty_like(V.col)

    # Fell the row and collumn mappings with updated values in p basis.
    for idx in range(len(V.row)):

        # Take unflatnes the V row indecies
        m1, m2 = divmod(V.row[idx], state)

        # Takes the associated indecies in p basis
        p1 = perm[m1]
        p2 = perm[m2]

        # Flatnes them again
        row_mapped[idx] = p1 * state + p2

        # Take unflatnes the V collumn indecies
        m1p, m2p = divmod(V.col[idx], state)

        # Takes the associated indecies in p basis
        pp1 = perm[m1p]
        pp2 = perm[m2p]

        # Flatnes them again
        col_mapped[idx] = pp1 * state + pp2

    # Rerturns a CSR matrix with a mappings in p basis.
    return sp.csr_matrix((V.data, (row_mapped, col_mapped)), shape=V.shape)

def build_V(state: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Constructs two sparse operators for a two-particle rotor system:
    - K: a diagonal kinetic energy operator in the 'p' basis
    - V: a sparse Potential energy operator in the 'm' basis

    The interaction operator V is defined over a 2D tensor-product space of 
    dimension `state * state` with elements given by:
        V = 0.75 * (|m1, m2><m1+1, m2+1| + |m1, m2><m1-1, m2-1|)
            - 0.25 * (|m1, m2><m1+1, m2-1| + |m1, m2><m1-1, m2+1|)

    The matrix K is diagonal with entries corresponding to squared momenta
    from the p-basis vector: K = diag(p^2), where p = vector_in_p(state).

    Parameters
    ----------
    state : int
        Total number state in the system, counting the ground state. Ex: system of -1, 0, 1 would be a system of 3 state.

    Returns
    -------
    tuple[sp.csr_matrix, sp.csr_matrix]
        - K: diagonal kinetic energy operator as a sparse CSR matrix in the 'p' basis
        - V: sparse potential energy operator as a CSR matrix in the 'm' basis
    """

    dim_V = state * state  # Total size of 2-particle Hilbert space

    data_V = []
    rows = []
    cols = []

    # Flatten 2D indices (m1, m2) into a single index for matrix representation
    def index(m1: int, m2: int) -> int:
        return m1 * state + m2

    # Loop over all possible (m1, m2) combinations
    for m1 in range(state):
        for m2 in range(state):
            i = index(m1, m2)  # Row index

            # Term: <m1+1, m2+1|
            if m1 + 1 < state and m2 + 1 < state:
                j = index(m1 + 1, m2 + 1)
                rows.append(i)
                cols.append(j)
                data_V.append(0.75)

            # Term: <m1-1, m2-1|
            if m1 - 1 >= 0 and m2 - 1 >= 0:
                j = index(m1 - 1, m2 - 1)
                rows.append(i)
                cols.append(j)
                data_V.append(0.75)

            # Term: <m1+1, m2-1|
            if m1 + 1 < state and m2 - 1 >= 0:
                j = index(m1 + 1, m2 - 1)
                rows.append(i)
                cols.append(j)
                data_V.append(-0.25)

            # Term: <m1-1, m2+1|
            if m1 - 1 >= 0 and m2 + 1 < state:
                j = index(m1 - 1, m2 + 1)
                rows.append(i)
                cols.append(j)
                data_V.append(-0.25)

    # Construct the sparse matrix V using CSR format
    V = sp.csr_matrix((data_V, (rows, cols)), shape=(dim_V, dim_V), dtype=np.float64)

    # Build kinetic energy operator K as a diagonal matrix in the p basis
    p = vector_in_p(state)
    K = sp.diags(p**2, offsets=0, format='csr')

    return K, V

def build_V_non_zero(state: int) -> tuple[sp.coo_matrix, sp.coo_matrix]:
    """
    Constructs:
    - K: diagonal kinetic energy operator in the p-basis
    - V: sparse potential energy operator, built in m-basis and transformed to p-basis

    The interaction operator V is built using structured rotor interaction rules,
    and then converted from m-basis to p-basis using m_to_p().

    Parameters
    ----------
    state : int
        Number of m-states: e.g. m in [-2, -1, 0, 1, 2] → state = 5

    Returns
    -------
    tuple[sp.coo_matrix, sp.coo_matrix]
        - K: diagonal kinetic energy matrix in p-basis (coo_matrix)
        - V: sparse potential energy matrix in p-basis (coo_matrix)
    """
    dim_V = state * state

    data_V = []
    rows = []
    cols = []

    def index(m1: int, m2: int) -> int:
        return m1 * state + m2

    # === Build V in m-basis ===
    for x in range(state - 1):
        rows.append(index(x, x))
        cols.append(index(x + 1, x + 1))
        data_V.append(0.75)

        rows.append(index(x + 1, x + 1))
        cols.append(index(x, x))
        data_V.append(0.75)

        rows.append(index(x + 1, x))
        cols.append(index(x, x + 1))
        data_V.append(-0.25)

        rows.append(index(x, x + 1))
        cols.append(index(x + 1, x))
        data_V.append(-0.25)

        for y in range(state - x - 1):
            rows.append(index(x, x + y))
            cols.append(index(x + 1, x + 1 + y))
            data_V.append(0.75)

            rows.append(index(x, x + 1 + y))
            cols.append(index(x + 1, x + y))
            data_V.append(-0.25)

            rows.append(index(x + 1, x + y))
            cols.append(index(x, x + 1 + y))
            data_V.append(-0.25)

            rows.append(index(x + 1, x + 1 + y))
            cols.append(index(x, x + y))
            data_V.append(0.75)

            rows.append(index(x + y, x + 1))
            cols.append(index(x + 1 + y, x))
            data_V.append(-0.25)

            rows.append(index(x + 1 + y, x + 1))
            cols.append(index(x + y, x))
            data_V.append(0.75)

            rows.append(index(x + y, x))
            cols.append(index(x + 1 + y, x + 1))
            data_V.append(0.75)

            rows.append(index(x + 1 + y, x))
            cols.append(index(x + y, x + 1))
            data_V.append(-0.25)

    V_m = sp.coo_matrix((data_V, (rows, cols)), shape=(dim_V, dim_V), dtype=np.float64)

    # === Build K in p-basis ===
    p = vector_in_p(state)  # e.g., [0, -1, 1, -2, 2]
    K = sp.coo_matrix((p**2, (np.arange(state), np.arange(state))), shape=(state, state), dtype=np.float64)

    # === m → p transformation for V ===

    # Create m_vals (e.g. [-2, -1, 0, 1, 2])
    m_vals = np.arange(-(state - 1) // 2, (state - 1) // 2 + 1)

    # Build m_index → p_index map
    perm = np.vectorize(m_to_p)(m_vals)  # m_to_p must be defined

    row_mapped = np.empty_like(V_m.row)
    col_mapped = np.empty_like(V_m.col)

    for k in range(len(V_m.data)):
        m1, m2 = divmod(V_m.row[k], state)
        m3, m4 = divmod(V_m.col[k], state)

        p1 = perm[m1]
        p2 = perm[m2]
        p3 = perm[m3]
        p4 = perm[m4]

        row_mapped[k] = p1 * state + p2
        col_mapped[k] = p3 * state + p4

    V_p = sp.coo_matrix((V_m.data, (row_mapped, col_mapped)), shape=(dim_V, dim_V), dtype=np.float64)

    return K, V_p

def build_V_in_p(state: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
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
        for row_2s, col_2s, val in zip(V.row, V.col, V.data): # Iterate only through non-zero elements of the 
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