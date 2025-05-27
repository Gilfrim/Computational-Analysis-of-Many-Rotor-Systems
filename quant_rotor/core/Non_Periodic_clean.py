import numpy as np
import csv
from quant_rotor.models import functions as func
import quant_rotor.core.hamiltonian_Illia as h
from importlib.resources import files
from dataclasses import dataclass

@dataclass
class SimulationParams:
    a: int
    i: int
    p: int
    sites: int
    i_method: int
    gap: bool
    gap_site: int
    epsilon: np.ndarray

@dataclass
class TensorData:
    t_a_i_tensor: np.ndarray
    t_ab_ij_tensor: np.ndarray
    h_full: np.ndarray
    v_full: np.ndarray

def A_term(a_upper, a_site, tensors: TensorData):
    return np.hstack((-tensors.t_a_i_tensor[a_site], np.identity(a_upper)))

def B_term(b_lower, b_site, tensors: TensorData):
    return np.vstack((np.identity(b_lower), tensors.t_a_i_tensor[b_site]))

def h_term(h_upper, h_lower, tensors: TensorData, params: SimulationParams):
    a_h_shift = [params.i if a_check == params.a else 0 for a_check in (h_upper, h_lower)]
    return tensors.h_full[a_h_shift[0]:h_upper + a_h_shift[0], a_h_shift[1]:h_lower + a_h_shift[1]]

def v_term(v_upper_1, v_upper_2, v_lower_1, v_lower_2, v_site_1, v_site_2, tensors: TensorData, params: SimulationParams):
    if params.gap and ((v_site_1 == params.gap_site and v_site_2 == params.gap_site + 1) or
                       (v_site_1 == params.gap_site + 1 and v_site_2 == params.gap_site)):
        return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))
    if abs(v_site_1 - v_site_2) == 1:
        a_v_shift = [params.i if a_check == params.a else 0 for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)]
        return tensors.v_full[
            a_v_shift[0]:v_upper_1 + a_v_shift[0],
            a_v_shift[1]:v_upper_2 + a_v_shift[1],
            a_v_shift[2]:v_lower_1 + a_v_shift[2],
            a_v_shift[3]:v_lower_2 + a_v_shift[3]
        ]
    else:
        return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))

def t_term(t_site_1, t_site_2, tensors: TensorData):
    return tensors.t_ab_ij_tensor[t_site_1, t_site_2]

def residual_single(x_s:int, tensors: TensorData, params: SimulationParams)->np.array:
    """Calculates R^{a}_{i}(x) singles equation"""

    sites, i_method, a, i, p= params.sites, params.i_method, params.a, params.i, params.p

    R_single = np.zeros((a, i))

    R_single += np.einsum("ap, pq, qi->ai", A_term(a, x_s), h_term(p, p), B_term(i, x_s))
    for z_s in range(sites):
        if z_s != x_s:
            if i_method >= 1:
                # noinspection SpellCheckingInspection
                R_single += np.einsum("ap, plcd, cdil->ai", A_term(a, x_s), v_term(p, i, a, a, x_s, z_s), t_term(x_s, z_s))
            # noinspection SpellCheckingInspection
            R_single += np.einsum("ap, plqs, qi, sl->ai", A_term(a, x_s), v_term(p, i, p, p, x_s, z_s), B_term(i, x_s), B_term(i, z_s))

    return R_single

def residual_double_sym(x_d:int, y_d:int, tensors: TensorData, params: SimulationParams)->np.array:
    """Calculates Rs^{ab}_{ij}(x < y) symmetric doubles equation"""
    sites, i_method, p, i, a = params.sites, params.i_method, params.p, params.i, params.a

    R_double_symmetric = np.zeros((a, a, i, i))

    if i_method >= 1:
        # noinspection SpellCheckingInspection
        R_double_symmetric += np.einsum("ap, bq, pqrs, ri, sj->abij", A_term(a, x_d), A_term(a, y_d), v_term(p, p, p, p, x_d, y_d), B_term(i, x_d), B_term(i, y_d))
        if i_method >= 2:
            # noinspection SpellCheckingInspection
            R_double_symmetric += np.einsum("ap, bq, pqcd, cdij->abij", A_term(a, x_d), A_term(a, x_d), v_term(p, p, a, a, x_d, y_d), t_term(x_d, y_d))
            # noinspection SpellCheckingInspection
            R_double_symmetric -= np.einsum("abkl, klpq, pi, qj->abij", t_term(x_d, y_d), v_term(i, i, p, p, x_d, y_d), B_term(i, x_d), B_term(i, y_d))
            if i_method == 3:
                # noinspection SpellCheckingInspection
                R_double_symmetric -= np.einsum("abkl, klcd, cdij->abij", t_term(x_d, y_d), v_term(i, i, a, a, x_d, y_d), t_term(x_d, y_d))
                if sites >= 4:
                    for z_ds in range(sites):
                        for w_ds in range(sites):
                            if z_ds not in {x_d, y_d} and w_ds not in {x_d, y_d} and z_ds != w_ds:
                                # noinspection SpellCheckingInspection
                                R_double_symmetric += np.einsum("klcd, acik, bdjl->abij", v_term(i, i, a, a, z_ds, w_ds), t_term(x_d, z_ds), t_term(y_d, w_ds))

    return R_double_symmetric

def residual_double_non_sym_1(x_d:int, y_d:int, tensors: TensorData, params: SimulationParams)->np.array:
    """Calculates Rn^{ab}_{ij}(x, y) non-symmetric doubles equation"""

    sites, i_method, p, i, a = params.sites, params.i_method, params.p, params.i, params.a

    R_double_non_symmetric_1 = np.zeros((a, a, i, i))

    if i_method >= 1:
        # noinspection SpellCheckingInspection
        R_double_non_symmetric_1 += np.einsum("ap, pc, cbij->abij", A_term(a, x_d), h_term(p, a), t_term(x_d, y_d))
        # noinspection SpellCheckingInspection
        R_double_non_symmetric_1 -= np.einsum("abkj, kp, pi->abij", t_term(x_d, y_d), h_term(i, p), B_term(i, x_d))

        if i_method >= 2:
            for z_dns_1 in range(sites):
                if z_dns_1 != x_d and z_dns_1 != y_d:
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_1 += np.einsum("acik, krcs, br, sj->abij", t_term(x_d, z_dns_1), v_term(i, p, a, p, z_dns_1, y_d), A_term(a, y_d), B_term(i, y_d))
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_1 += np.einsum("bq, qlds, adij, sl->abij", A_term(a, y_d), v_term(p, i, a, p, y_d, z_dns_1), t_term(x_d, y_d), B_term(i, z_dns_1))
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_1 -= np.einsum("abkj, lkrp, pi, rl->abij", t_term(x_d, y_d), v_term(i, i, p, p, z_dns_1, x_d), B_term(p, i, x_d), B_term(i, z_dns_1))

    return R_double_non_symmetric_1

def residual_double_non_sym_2(x_d:int, y_d:int, tensors: TensorData, params: SimulationParams)->np.array:
    """Calculates Rn^{ba}_{ji}(y, x) ai -> jb permutation non-symmetric doubles equation"""

    sites, i_method, p, i, a = params.sites, params.i_method, params.p, params.i, params.a

    R_double_non_symmetric_2 = np.zeros((a, a, i, i))

    if i_method >= 1:
        # noinspection SpellCheckingInspection
        R_double_non_symmetric_2 += np.einsum("bp, pc, caji->baji", A_term(a, y_d), h_term(p, a), t_term(y_d, x_d))
        # noinspection SpellCheckingInspection
        R_double_non_symmetric_2 -= np.einsum("baki, kp, pj->baji", t_term(y_d, x_d), h_term(i, p), B_term(i, y_d))

        if i_method >= 2:
            for z_dns_2 in range(sites):
                if z_dns_2 != x_d and z_dns_2 != y_d:
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_2 += np.einsum("bcjk, krcs, ar, si->baji", t_term(y_d, z_dns_2), v_term(i, p, a, p, z_dns_2, x_d), A_term(a, x_d), B_term(i, x_d))
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_2 += np.einsum("aq, qlds, bdji, sl->baji", A_term(a, x_d), v_term(p, i, a, p, x_d, z_dns_2), t_term(y_d, x_d), B_term(i ,z_dns_2))
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_2 -= np.einsum("baki, lkrp, pj, rl->baji", t_term(y_d, x_d), v_term(i, i, p, p, z_dns_2, y_d), B_term(i, y_d), B_term(i ,z_dns_2))

    return R_double_non_symmetric_2

def residual_double_total(x_d:int, y_d:int, tensors: TensorData, params: SimulationParams)->np.array:
    """Calculates Rt^{ab}_{ij}(x < y) = Rs^{ab}_{ij}(x < y) + Rn^{ab}_{ij}(x < y) + Rn^{ba}_{ji}(y < x)total doubles equation"""

    return residual_double_sym(x_d, y_d) + residual_double_non_sym_1(x_d, y_d) + residual_double_non_sym_2(x_d, y_d)


def update_one(r_1_value, params: SimulationParams):
    a, i, eps = params.a, params.i, params.epsilon
    update = np.zeros((a, i))
    for u_a in range(a):
        for u_i in range(i):
            update[u_a, u_i] = 1 / (eps[u_a + i] - eps[u_i])
    return update * r_1_value

def update_two(r_2_value, params: SimulationParams):
    a, b, i, j, eps = params.a, params.b, params.i, params.j, params.epsilon
    update = np.zeros((a, b, i, j))
    for u_a in range(a):
        for u_b in range(b):
            for u_i in range(i):
                for u_j in range(j):
                    update[u_a, u_b, u_i, u_j] = 1 / (eps[u_a + i] + eps[u_b + j] - eps[u_i] - eps[u_j])
    return update * r_2_value

def save_t_1_amplitudes(iteration, tensors: TensorData):
    np.save(f"t1_amplitudes_iter_{iteration}.npy", tensors.t_a_i_tensor)

def save_t_2_amplitudes(iteration, tensors: TensorData):
    np.save(f"t2_amplitudes_iter_{iteration}.npy", tensors.t_ab_ij_tensor)

def run_simulation(
    sites: int,
    states: int,
    low_states: int,
    initial: float,
    threshold: float,
    g: float,
    i_method: int,
    gap: bool,
    gap_site: int,
    transform: bool
):
    """
    Direct input version of the simulation setup.

    Parameters:
    - sites (int): Number of lattice sites.
    - states (int): Number of rotor states.
    - low_states (int): Cutoff for low-energy states.
    - initial (float): Initial value for the solver.
    - threshold (float): Convergence threshold.
    - g (float): Coupling constant.
    - i_method (int): Method selection index.
    - gap (bool): Whether to compute energy gap.
    - gap_site (int): Lattice site index for gap measurement (will be decremented).
    - transform (bool): Whether to apply a transformation.

    Returns:
    - dict: Placeholder output dictionary. Replace with your simulation return.
    """

    gap_site = gap_site - 1  # adjust index if needed

    #state variables
    #could just use p, i, a
    #makes checking einsums and such a bit easier
    p = states
    i = low_states
    a = p - i

    #checks for i and a sharing same state level
    if i %2 == 0 or a %2 != 0:
        raise ValueError("Overlap between low and high stats, will cause divide by zero in denominator update")

    m_max = abs(func.p_to_m(states - 1))
    m_max_shaeer = 5

    #for states > 11 would need to have shaeers code generate csv files with m_max > 5
    #gets h values from csv files made with Shaeer's code and puts them into a dictionary

    # Load .npy matrices directly from the package
    data_dir = files("quant_rotor.data")
    K = np.load(data_dir / "K_matrix.npy")
    V = np.load(data_dir / "V_matrix.npy")

    V = V + V.T - np.diag(np.diag(V))
    V_tensor = V.reshape(p, p, p, p)  # Adjust if needed

    h_full = h.basis_m_to_p_matrix_conversion(K)
    v_full = h.basis_m_to_p_matrix_conversion(V_tensor)

    #t1 and t2 amplitude tensors
    t_a_i_tensor = np.full((sites, a, i), initial)
    t_ab_ij_tensor = np.full((sites, sites, a, a, i, i), initial)

    params = SimulationParams(
    a=a,
    i=i,
    p=p,  # These can be the same as `a + i` or chosen independently
    sites=sites,
    i_method=i_method,
    gap=gap,
    gap_site=gap_site,
    epsilon=epsilon
    )

    tensors = TensorData(
    t_a_i_tensor=t_a_i_tensor,
    t_ab_ij_tensor=t_ab_ij_tensor,
    h_full=h_full,
    v_full=v_full
    )

    del K
    

    #eigenvalues from h for update
    epsilon = np.diag(tensors.h_full)

    #unitary transformation test
    #same as transformation test.py
    if transform:
        h_off = tensors.h_full.copy()

        for p_off in range(states):
            for q_off in range(states):
                if p_off != q_off:
                    h_off[p_off, q_off] = 0.1 / abs(p_off - q_off)

        _, U = np.linalg.eigh(h_off)
        tensors.h_full = U.T @ h_full @ U

        # noinspection SpellCheckingInspection
        v_full = np.einsum("ip, jr, prqs, qk, sl->ijkl", U.T, U.T, v_full, U, U)

        file_path_energy = "energy.csv"

        with open(file_path_energy, "w", encoding="utf-8") as energy_file:
            energy_file.write("Iteration, Energy, ΔEnergy\n")

            iteration = 0
            single = np.zeros((sites, a, i))
            double = np.zeros((sites, sites, a, a, i, i))
            previous_energy = 0

            while True:
                energy = 0

                for x_site in range(sites):
                    single[x_site] = residual_single(x_site)
                    for y_site in range(sites):
                        if x_site < y_site:
                            double[x_site, y_site] = residual_double_total(x_site, y_site)

                print(f"1 max: {np.max(np.abs(single))}")
                print(f"2 max: {np.max(np.abs(double))}")

                if np.all(abs(single) <= threshold) and np.all(abs(double) <= threshold):
                    break

                if np.isnan(single).any() or np.isnan(double).any() or np.isinf(single).any() or np.isinf(double).any():
                    raise ValueError("Diverges (inf or nan)")

                # Save amplitudes as NumPy arrays
                save_t_1_amplitudes(iteration, t_a_i_tensor)
                save_t_2_amplitudes(iteration, t_ab_ij_tensor)

                for site_u_1 in range(sites):
                    t_a_i_tensor[site_u_1] -= update_one(single[site_u_1])
                    for site_u_2 in range(sites):
                        if site_u_1 < site_u_2:
                            t_ab_ij_tensor[site_u_1, site_u_2] -= update_two(double[site_u_1, site_u_2])

                for site_x in range(sites):
                    energy += np.einsum("ip, pi->", h_term(i, p), B_term(i, site_x))
                    for site_y in range(sites):
                        if site_x < site_y:
                            energy += np.einsum("ijab, abij->", v_term(i, i, a, a, site_x, site_y), t_term(site_x, site_y))
                            energy += np.einsum("ijpq, pi, qj->", v_term(i, i, p, p, site_x, site_y), B_term(i, site_x), B_term(i, site_y))

                delta_energy = energy - previous_energy
                previous_energy = energy

                iteration += 1
                print(f"Iteration #: {iteration}")
                print(f"Energy: {float(energy)}")

                energy_file.write(f"{iteration}, {energy}, {delta_energy}\n")



if __name__ == "__main__":
    run_simulation(5, 5, 1, 0, 1e-8, 0.1, 0, True, 3, True)
    