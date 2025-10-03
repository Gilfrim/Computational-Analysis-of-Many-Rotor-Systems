import numpy as np
import scipy.sparse as sp

from quant_rotor.models.sparse.support_ham import (
    H_kinetic_sparse,
    H_potential_sparse,
    build_V_in_p,
)


def hamiltonian_sparse(state: int, site: int, g_val: float, tau: float=0, l_val: float=0, K_import: sp.csr_matrix=[], V_import: sp.csr_matrix=[], Import: bool=False, spar: bool=False, general: bool=False)->tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
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
    l_val : float, optional
        A multiplier for the kinetic energy. Creates a tridiagonal matrix with zeros on the
        diagonal and l_val / sqrt(pi) on the off-diagonals. Defaults to 0 (no modification).
    K_import : sp.csr_matrix, optional
        Predefined kinetic energy matrix in the p-basis, shape (state, state).
        If provided, this skips the construction and basis conversion of the kinetic matrix.
        Defaults to None or empty list.
    V_import : sp.csr_matrix, optional
        Predefined potential energy matrix in the p-basis, shape (state² * state²).
        If provided, this skips construction, symmetrization, reshaping, and basis conversion
        of the potential matrix. Defaults to None or empty list.
    Import : bool, optional
        If True, the function assumes both K_import and V_import are provided and skips
        generation of K and V from scratch. Must be True if using imported matrices.
        Defaults to False.

    Returns
    -------
    tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]
        Returns a tuple of three arrays in the following order:
        - Dense Hamiltonian matrix of shape (state^site, state^site), constructed from K and V
        - Kinetic energy matrix in p-basis, shape (state, state)
        - Potential energy matrix in p-basis, shape (state, state, state, state), symmetric
    """
    # Check if you are importing a Kinetic and Potential energy matrix or creating one from the scratch.
    if Import == False:
        # Create a Kinetic and Potential energy matricies.
        K_in_p, V_in_p = build_V_in_p(state, tau)

    else:
        # In case of importing skiping ass of the steps and assumes the Kinetic and Potential energy matricies are in the coreect shape and in p basis.
        K_in_p = K_import
        V_in_p = V_import

    # Construct a Kinetic and Potential hamiltonian.
    K_final = H_kinetic_sparse(state, site, K_in_p)
    V_final = H_potential_sparse(state, site, V_in_p, g_val)

    # Add to get the final hamiltonian.
    H_final = K_final + V_final

    # It is importatnt to keep the return in this format since hamiltonian_big.py functions are using this structure.
    return H_final, K_in_p, V_in_p
