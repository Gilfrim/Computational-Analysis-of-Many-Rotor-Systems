import matplotlib.pyplot as plt
import numpy as np
import opt_einsum as oe
import scipy as sp
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from quant_rotor.core.dense.de_solve_one_thermal import integration_scheme
from quant_rotor.core.dense.de_solve_one_thermal_dense import (
    integration_scheme as integration_scheme_fast,
)
from quant_rotor.core.dense.hamiltonian import hamiltonian_dense
from quant_rotor.core.dense.hamiltonian_big import (
    hamiltonian_big_dense,
    hamiltonian_general_dense,
)
from quant_rotor.core.dense.t_amplitudes_guess import (
    t_1_amplitude_guess_ground_state,
    t_2_amplitude_guess_ground_state,
)
from quant_rotor.core.dense.t_amplitudes_periodic import t_periodic
from quant_rotor.core.sparse.hamiltonian import hamiltonian_sparse
from quant_rotor.models.dense.density_matrix import density_matrix_1
from quant_rotor.models.dense.support_ham import (
    basis_m_to_p_matrix_conversion,
    m_to_p,
    write_matrix_elements,
)


def curve(D):

    m = ((D.shape)[0] - 1) // 2
    gama = np.arange((-np.pi / 2), (np.pi * 3 / 2) + 2 * np.pi / 100, 2 * np.pi / 50)

    vector_in_m = np.arange(-m, m + 1)

    index_map_m_to_p = np.vectorize(m_to_p)(vector_in_m)
    index_maps = [index_map_m_to_p] * 2

    D_in_m = D[np.ix_(*index_maps)]

    e_plus = np.exp(1j * np.outer(vector_in_m, gama))
    e_mines = np.exp(-1j * np.outer(vector_in_m, gama))

    y = oe.contract("ap, pq, qb->ab", e_mines.T, D_in_m, e_plus)

    return np.diag(y), gama


def calculate_xy(i):
    """Your calculation per frame. Modify this part only."""

    site = 2
    state = 21
    state_NO = i + 4

    g = 1
    D_val = 1e-4
    lambda_val = -1e-2

    periodic = False

    H_K_V = hamiltonian_dense(
        state, site, g, D=D_val, lambda_val=lambda_val, field=True, periodic=periodic
    )
    H, K, V = hamiltonian_dense(state, site, g, D=D_val, periodic=periodic)

    H_L, K_L, V_L = H_K_V[0], H_K_V[1], H_K_V[2]

    eig_val_L, eig_vec_L = np.linalg.eigh(H_L)

    D = density_matrix_1(state, site, eig_vec_L[:, np.argmin(eig_val_L)], 0)

    eig_val_D, matrix_p_to_NO_full = np.linalg.eigh(D)
    index_d = np.argsort(-eig_val_D)
    matrix_p_to_NO = matrix_p_to_NO_full[:, index_d[:state_NO]]

    K_NO = matrix_p_to_NO.T.conj() @ K @ matrix_p_to_NO
    matrix_p_to_NO_V = np.kron(matrix_p_to_NO, matrix_p_to_NO)
    V_NO = matrix_p_to_NO_V.conj().T @ V @ matrix_p_to_NO_V

    K_loc = matrix_p_to_NO @ K_NO @ matrix_p_to_NO.T.conj()
    V_loc = matrix_p_to_NO_V @ V_NO @ matrix_p_to_NO_V.conj().T

    H_loc, _, _ = hamiltonian_dense(
        state_NO,
        site,
        1,
        K_import=K_loc,
        V_import=V_loc,
        Import_K_V=True,
        periodic=periodic,
    )
    eig_val_loc, eig_vec_loc = np.linalg.eigh(H_loc)

    D_loc = density_matrix_1(state_NO, site, eig_vec_loc[:, np.argmin(eig_val_loc)], 0)
    y, x = curve(D_loc)

    return x, y


fig, ax = plt.subplots()
(line,) = ax.plot([], [], lw=2)
ax.set_xlim(-2.5, 5)  # adjust as needed
ax.set_ylim(0, 4)  # adjust as needed
ax.set_xlabel("x")
ax.set_ylabel("y")


def update(i):
    x, y = calculate_xy(i)
    line.set_data(x, y)
    return (line,)


frames = 15

ani = FuncAnimation(fig, update, frames=frames, interval=15, blit=True)

ani.save("animation.mp4", writer=FFMpegWriter(fps=10, bitrate=2000, codec="h264"))

# ------- SAVE as GIF (no ffmpeg needed; needs pillow) -------
# pip install pillow
# ani.save("animation.gif", writer=PillowWriter(fps=10))

plt.close(fig)  # optional: don’t also show a live window
print("Saved animation.mp4")
