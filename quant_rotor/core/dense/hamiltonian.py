import numpy as np

from quant_rotor.models.dense.support_ham import (
    H_kinetic,
    H_potential,
    basis_m_to_p_matrix_conversion,
    write_matrix_elements,
)


def hamiltonian_dense(
    state: int,
    site: int,
    g_val: float,
    psi_twist: float = 0,
    periodic: bool = True,
    l_val: float = 0,
    K_import: np.ndarray = [],
    V_import: np.ndarray = [],
    Import: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs the Kinetic, Potential, and dense Hamiltonian operators for a specified system,
    described by the number of states, sites, and g-value modifier to the potential energy.

    Alternatively, if Kinetic and Potential energy matrices are imported, constructs a
    Hamiltonian from them.

    Parameters
    ----------
    state : int
        Total number states in the system, counting the ground state. Ex: system of -1, 0, 1 would be a system of 3 states.
    site : int
        The number of rotors (sites) in the system.
    g_val : float
        The constant multiplier for the potential energy. Typically in the range 0 <= g <= 1.
    tau : float, optional
        Dipolar plains chain angle.
        Defaults to complanar or 0.
    periodic : bool
        Defines if the hamiltonian for the periodic system or the non-peirodic system.
        Defaults to True.
    l_val : float, optional
        A multiplier for the kinetic energy. Creates a tridiagonal matrix with zeros on the
        diagonal and l_val / sqrt(pi) on the off-diagonals. Defaults to 0 (no modification).
    K_import : np.ndarray, optional
        Predefined kinetic energy matrix in the p-basis, shape (state, state).
        If provided, this skips the construction and basis conversion of the kinetic matrix.
        Defaults to None or empty list.
    V_import : np.ndarray, optional
        Predefined potential energy matrix in the p-basis, shape (state² * state²).
        If provided, this skips construction, symmetrization, reshaping, and basis conversion
        of the potential matrix. Defaults to None or empty list.
    Import : bool, optional
        If True, the function assumes both K_import and V_import are provided and skips
        generation of K and V from scratch. Must be True if using imported matrices.
        Defaults to False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Returns a tuple of three arrays in the following order:
        - Dense Hamiltonian matrix of shape (state^site, state^site), constructed from K and V
        - Kinetic energy matrix in p-basis, shape (state, state)
        - Potential energy matrix in p-basis, shape (state, state, state, state), symmetric
    """
    # Check if you are importing a Kinetic and Potential energy matrix or creating one from the scratch.
    if Import == False:

        # Create a Kinetic and Potential energy matricies.
        K, V = write_matrix_elements((state - 1) // 2, psi_twist)

        # Optional modifier for the one body (Kinetic energy) operator.
        # Creates and adds a tridiagonal matrix with 0 along the center diagonal and two shifted diagonals determened
        # by the value "l" to the Kinetic energy matrix.
        # l = l_val * 1/np.sqrt(2)
        # L_sparce = diags([l, 0, l], offsets=[-1, 0, 1], shape=(state, state))
        # L_dense = L_sparce.toarray()
        # K = K + L_dense

        # Reshape a potential energy matrix from (state^2, state^2) -> (state, state, state, state).
        V_tensor = V.reshape(state, state, state, state)

        # Transform Kinnetic and Potential energy matricies from m basis to p basis.
        K_in_p = basis_m_to_p_matrix_conversion(K, state)
        V_in_p = basis_m_to_p_matrix_conversion(V_tensor, state)

        # Reshape a Potential energy matrix back from (state, state, state, state) -> (state^2, state^2).
        V_in_p = V_in_p.reshape(state**2, state**2) * g_val

    else:
        # In case of importing skiping ass of the steps and assumes the Kinetic and Potential energy matricies are in the coreect shape and in p basis.
        K_in_p = K_import
        V_in_p = V_import

    # Construct a Kinetic and Potential hamiltonian.
    K_final = H_kinetic(state, site, K_in_p)
    V_final = H_potential(state, site, V_in_p, 1, periodic)

    # Add to get the final hamiltonian.
    H_final = K_final + V_final

    # It is importatnt to keep the return in this format since hamiltonian_big.py functions are using this structure.
    return H_final, K_in_p, V_in_p
