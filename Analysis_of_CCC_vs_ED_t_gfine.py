import os
import sys

import numpy as np

sys.path.append(
    "/Users/gilfrim/Desktop/Projects/Computational_Analysis_of_Many_Rotor_Systems/"
)

from quant_rotor.CCC_integration_methods.Dense.de_solve_one_thermal_dense import (
    integration_scheme as integration_scheme_fast,
)
from quant_rotor.Hamiltonian_models.Sparse.hamiltonian_big import (
    hamiltonian_general_sparse,
)

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


# ======================
# Parameters
# ======================

state_prim = 11
state_on = 5
site = 5

periodic = True

g_fine = np.round(np.arange(0.05, 1.51, 0.05), 2)
angle_values = np.array([30])

t_final = 10
residual_tol = 1e-5

save_name = "Rotor_5site_ON5_CCC_t_amp_g_fine.npz"


# ======================
# Helper functions
# ======================


def get_available_save_path(base_name):
    save_dir = os.path.dirname(__file__) if "__file__" in globals() else "."
    base_path = os.path.join(save_dir, base_name)

    if not os.path.exists(base_path):
        return base_path

    name, ext = os.path.splitext(base_name)

    version = 1
    while True:
        new_name = f"{name}_v{version}{ext}"
        new_path = os.path.join(save_dir, new_name)

        if not os.path.exists(new_path):
            return new_path

        version += 1


def get_angle_label(angle_deg):
    angle_deg = float(angle_deg)

    if np.isclose(angle_deg, int(angle_deg)):
        return f"angle_{int(angle_deg)}"

    return f"angle_{angle_deg:g}"


def final_abs_value(x):
    arr = np.asarray(x)

    if arr.ndim == 0:
        return np.abs(arr.item())

    return np.abs(arr[-1])


def safe_ratio(max_val, norm_val):
    if np.abs(norm_val) < 1e-14:
        return np.nan

    return max_val / norm_val


def get_ccc_result(g_val, angle_deg):
    tau_rad = np.deg2rad(angle_deg)

    _, K_matrix, V_matrix = hamiltonian_general_sparse(
        state_prim,
        state_on,
        site,
        g_val,
        tau=tau_rad,
    )

    K_dense = K_matrix.toarray() if hasattr(K_matrix, "toarray") else K_matrix
    V_dense = V_matrix.toarray() if hasattr(V_matrix, "toarray") else V_matrix

    if V_dense.ndim == 2:
        V_dense = V_dense.reshape(state_on, state_on, state_on, state_on)

    i_state = 1
    a_state = state_on - i_state

    t_1_tensor = np.zeros((site, a_state), dtype=complex)
    t_2_tensor = np.zeros((site, site, a_state, a_state), dtype=complex)

    t_0_import = np.log(state_on**site)

    time_array, energy_array, t_0_CCC, t_1_out, t_2_out = integration_scheme_fast(
        site,
        state_on,
        K_dense,
        V_dense,
        V_dense,
        periodic=periodic,
        import_guess=True,
        t_0_import=t_0_import,
        t_1_import=t_1_tensor,
        t_2_import=t_2_tensor,
        t_final=t_final,
    )

    E_CCC = energy_array[-1].real / site

    final_residual = 0
    converged = True
    n_step = len(time_array)

    max_t1 = float(np.max(np.abs(t_1_out)))
    max_t2 = float(np.max(np.abs(t_2_out)))

    norm_t1 = float(np.linalg.norm(t_1_out.ravel()))
    norm_t2 = float(np.linalg.norm(t_2_out.ravel()))

    ratio_t1 = safe_ratio(max_t1, norm_t1)
    ratio_t2 = safe_ratio(max_t2, norm_t2)

    ratio_t1_scaled = ratio_t1 * g_val
    ratio_t2_scaled = ratio_t2 * g_val

    return (
        E_CCC,
        final_residual,
        converged,
        n_step,
        max_t1,
        max_t2,
        norm_t1,
        norm_t2,
        ratio_t1,
        ratio_t2,
        ratio_t1_scaled,
        ratio_t2_scaled,
    )


# ======================
# Main
# ======================


def run_rotor_ccc_t_amp_g_fine_batch():
    save_dict = {
        "state_prim": state_prim,
        "state_on": state_on,
        "site": site,
        "periodic": periodic,
        "g_fine": g_fine,
        "angle_values": angle_values,
        "t_final": t_final,
        "residual_tol": residual_tol,
        "basis": "ON_basis",
        "note": "Outer loop: angle_values. Inner loop: g_fine. E_CCC is saved per site.",
    }

    total_tasks = len(angle_values) * len(g_fine)
    task_id = 0

    for angle_deg in angle_values:
        label = get_angle_label(angle_deg)

        print("\n==============================")
        print(f"angle = {angle_deg:.1f}")
        print("==============================")

        E_CCC_list = []
        final_residual_list = []
        converged_list = []
        n_step_list = []

        max_t1_list = []
        max_t2_list = []
        norm_t1_list = []
        norm_t2_list = []

        ratio_t1_list = []
        ratio_t2_list = []
        ratio_t1_scaled_list = []
        ratio_t2_scaled_list = []

        for g_val in g_fine:
            task_id += 1

            print("\n------------------------------")
            print(
                f"[{task_id:03d}/{total_tasks:03d}] angle = {angle_deg:.1f}, g = {g_val:.2f}"
            )
            print("------------------------------")

            try:
                (
                    E_CCC,
                    final_residual,
                    converged,
                    n_step,
                    max_t1,
                    max_t2,
                    norm_t1,
                    norm_t2,
                    ratio_t1,
                    ratio_t2,
                    ratio_t1_scaled,
                    ratio_t2_scaled,
                ) = get_ccc_result(g_val, angle_deg)

                print(f"E_CCC/site = {E_CCC}")
                print(f"final residual = {final_residual:.6e}")
                print(f"converged = {converged}")
                print(f"n_step = {n_step}")
                print(f"max |t1| = {max_t1:.6e}")
                print(f"max |t2| = {max_t2:.6e}")
                print(f"norm t1 = {norm_t1:.6e}")
                print(f"norm t2 = {norm_t2:.6e}")
                print(f"max |t2| / norm t2 = {ratio_t2:.6e}")
                print(f"g * max |t2| / norm t2 = {ratio_t2_scaled:.6e}")

                E_CCC_list.append(E_CCC)
                final_residual_list.append(final_residual)
                converged_list.append(converged)
                n_step_list.append(n_step)

                max_t1_list.append(max_t1)
                max_t2_list.append(max_t2)
                norm_t1_list.append(norm_t1)
                norm_t2_list.append(norm_t2)

                ratio_t1_list.append(ratio_t1)
                ratio_t2_list.append(ratio_t2)
                ratio_t1_scaled_list.append(ratio_t1_scaled)
                ratio_t2_scaled_list.append(ratio_t2_scaled)

            except Exception as e:
                print(f"error(angle={angle_deg}, g={g_val}): {e}")

                E_CCC_list.append(np.nan)
                final_residual_list.append(np.nan)
                converged_list.append(False)
                n_step_list.append(-1)

                max_t1_list.append(np.nan)
                max_t2_list.append(np.nan)
                norm_t1_list.append(np.nan)
                norm_t2_list.append(np.nan)

                ratio_t1_list.append(np.nan)
                ratio_t2_list.append(np.nan)
                ratio_t1_scaled_list.append(np.nan)
                ratio_t2_scaled_list.append(np.nan)

        save_dict[f"E_CCC_{label}"] = np.array(E_CCC_list)
        save_dict[f"final_residual_{label}"] = np.array(final_residual_list)
        save_dict[f"converged_{label}"] = np.array(converged_list)
        save_dict[f"n_step_{label}"] = np.array(n_step_list)

        save_dict[f"max_t1_{label}"] = np.array(max_t1_list)
        save_dict[f"max_t2_{label}"] = np.array(max_t2_list)
        save_dict[f"norm_t1_{label}"] = np.array(norm_t1_list)
        save_dict[f"norm_t2_{label}"] = np.array(norm_t2_list)

        save_dict[f"ratio_t1_{label}"] = np.array(ratio_t1_list)
        save_dict[f"ratio_t2_{label}"] = np.array(ratio_t2_list)
        save_dict[f"ratio_t1_scaled_{label}"] = np.array(ratio_t1_scaled_list)
        save_dict[f"ratio_t2_scaled_{label}"] = np.array(ratio_t2_scaled_list)

        save_path_partial = get_available_save_path(
            f"Rotor_5site_ON5_CCC_t_amp_g_fine_partial_{label}.npz"
        )

        np.savez(save_path_partial, **save_dict)
        print(f"\npartial saved to\n{save_path_partial}")

    save_path = get_available_save_path(save_name)

    np.savez(save_path, **save_dict)
    print(f"\nsaved to\n{save_path}")


run_rotor_ccc_t_amp_g_fine_batch()
