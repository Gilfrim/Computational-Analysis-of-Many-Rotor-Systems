import numpy as np
import os
from quant_rotor.models.support_ham import write_matrix_elements, basis_m_to_p_matrix_conversion, H_kinetic, H_potential

np.set_printoptions(suppress = True, linewidth = 1500, threshold = 10000, precision = 9)

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