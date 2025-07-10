import numpy as np
from quant_rotor.models.dense.support_ham import write_matrix_elements, basis_m_to_p_matrix_conversion, H_kinetic, H_potential, H_kinetic_sparse, H_potential_sparse, H_potential_general
from scipy.sparse import diags, lil_matrix

def hamiltonian_dence(state: int, site: int, g_val: float, l_val: float=0, K_import: np.ndarray=[], V_import: np.ndarray=[], Import: bool=False, spar: bool=False, general: bool=False)->np.ndarray:
    """_summary_

    Parameters
    ----------
    state : int
        Number of unique states in the system, not counting the ground state. Ex: system of -1, 0, 1 would be a system of 1
    site : int
        The number of rotors in the system.
    g_val : float
        The constant multiplier for the potential energy. Usually in the range of 0 <= g <= 1.
    l_val : float, optional
        A multiplier for the kinetic energy created a tridiagonal matrix with zeros on the diagonal and l-val / sqrt(pi) on two off-diagonal elements.  By default, 0, which means no modification of 
    K_import : np.ndarray, optional
        For importing the Kinetic Energy matrix. The matrix should be in basis p and dimension of (state, state). If you import the matrix, you skip: creation of the K matrix, conversion from n to p basis. By default, [].
    V_import : np.ndarray, optional
      	For importing the Potential Energy matrix. The matrix should be in basis p and shape of (state^2, state^2). If you import the matrix, you skip: creation of the V; symmetrizing V; reshaping V to a tensor of shape (state, state, state, state); conversion from n to p basis. By default, [].
    Import : bool, optional
        Condition: if you are importing the Kinetic and Potential energy matrices should be set to True. You can only import Kinetic and Potential together. By default, False.

    Returns
    -------
    np.ndarray
        A complete, dence, Hamiltonian of shape (state^site, state^site).
    """
    if Import == False:
        K, V = write_matrix_elements((state-1) // 2)

        # L_sparce = diags([l_val * 1/np.sqrt(2), 0, l_val * 1/np.sqrt(2)], offsets=[-1, 0, 1], shape=(state, state))

        # L_dense = L_sparce.toarray()

        # K = K + L_dense

        V_from_npy = V + V.T - np.diag(np.diag(V))
        V_tensor = V_from_npy.reshape(state, state, state, state)

        K_in_p = basis_m_to_p_matrix_conversion(K)
        V_in_p = basis_m_to_p_matrix_conversion(V_tensor)

        V_in_p = V_in_p.reshape(state**2, state**2)

    else:
        K_in_p = K_import
        V_in_p = V_import

    if spar:
        K_in_p = lil_matrix(K_in_p)
        V_in_p = lil_matrix(V_in_p)
        K_final = H_kinetic_sparse(state, site, K_in_p)
        V_final = H_potential_sparse(state, site, V_in_p, g_val)
    elif general:
        K_in_p = lil_matrix(K_in_p)
        V_in_p = lil_matrix(V_in_p)
        K_final = H_kinetic(state, site, K_in_p)
        V_final = H_potential_general(state, site, V_in_p, g_val)
    else:
        K_final = H_kinetic(state, site, K_in_p)
        V_final = H_potential(state, site, V_in_p, g_val)

    H_final = K_final + V_final

    return H_final, K_in_p, V_in_p

if __name__ == "__main__":
    site = 3
    state = 3
    print(hamiltonian_dence(state, site))