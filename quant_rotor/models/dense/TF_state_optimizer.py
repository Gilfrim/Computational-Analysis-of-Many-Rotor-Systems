import matplotlib.pyplot as plt
import numpy as np
import opt_einsum as oe
from scipy.optimize import minimize

import quant_rotor.models.dense.thermofield_boltz_funcs as bz
from quant_rotor.core.dense.de_solve_one_thermal import integration_scheme
from quant_rotor.core.dense.hamiltonian import hamiltonian_dense
from quant_rotor.core.dense.t_amplitudes_guess import amplitute_energy
from quant_rotor.models.dense.density_matrix import density_matrix_1


def f_max(C, ground_NO, state_2, site):
    # C is variable; ground_NO is fixed from args
    psi = ground_NO @ C
    D = density_matrix_1(state_2, site, psi, 0)
    return float(np.real(np.trace(D @ D)))


def constr_ineq(C, ground_NO, state_2, site):
    return 1 - f_max(C, ground_NO, state_2, site)


# optional equality constraint: <psi|psi> = 1
def constr_eq(C):
    return 1 - float(C @ C)


def d_squared_overlap_basis(theta: float, C_new: np.ndarray, psi, state, state_2, site):
    deg_state_d_2 = np.zeros((state**site - 1), dtype=complex)

    for i in range(state**site - 1):

        psi_i = np.cos(theta) * psi + np.sin(theta) * C_new[:, i]

        rho_site_0 = density_matrix_1(state_2, site, psi_i, 0)

        deg_state_d_2[i] = 1 - np.trace(rho_site_0 @ rho_site_0)

    return deg_state_d_2


def gradient_hassian(
    deg_ground_basis: np.ndarray, psi: np.ndarray, theta: float, state, state_2, site
):

    C_lambda = deg_ground_basis - np.outer(psi, psi @ deg_ground_basis)
    S = C_lambda @ C_lambda.T
    eig_val_S, eig_vec_S = np.linalg.eigh(S)

    index_non_zero = np.where(np.abs(eig_val_S) > 1e-5)[0]

    C_new = eig_vec_S[:, index_non_zero] * eig_val_S[index_non_zero] ** (-0.5)

    if not (
        np.allclose(np.ones(state**site - 1), np.diag((C_new.T @ C_new)), atol=1e-15)
    ):
        raise ValueError("Basis is not orthonormal.")

    g_i = (
        (
            d_squared_overlap_basis(theta, C_new, psi, state, state_2, site)
            - d_squared_overlap_basis(-theta, C_new, psi, state, state_2, site)
        )
        / 2
        * np.abs(theta)
    )

    h_i = (
        2 * d_squared_overlap_basis(0, C_new, psi, state, state_2, site)
        - d_squared_overlap_basis(theta, C_new, psi, state, state_2, site)
        - d_squared_overlap_basis(-theta, C_new, psi, state, state_2, site)
    ) / theta**2

    return g_i, h_i, C_new


def func_psi_u(deg_ground_basis, psi, theta, state, state_2, site):

    g_i, h_i, C_new = gradient_hassian(
        deg_ground_basis, psi, theta, state, state_2, site
    )
    g_norm = g_i / np.linalg.norm(g_i)

    if not (np.allclose(1, g_norm @ g_norm, atol=1e-15)):
        print(g_norm)
        # raise ValueError("Jacobian is not normalized.")
        raise ValueError(g_norm)

    psi_u = C_new @ g_norm

    return psi_u


def func_to_minmize(
    theta: float, psi: np.ndarray, deg_ground_basis: np.ndarray, state, state_2, site
):

    psi_int = np.cos(theta) * psi + np.sin(theta) * func_psi_u(
        deg_ground_basis, psi, theta, state, state_2, site
    )

    D_psi_int = density_matrix_1(state_2, site, psi_int, 0)

    return 1 - np.trace(D_psi_int @ D_psi_int)


def min_output(
    theta: float, psi: np.ndarray, deg_ground_basis: np.ndarray, state, state_2, site
):

    psi_int = np.cos(theta) * psi + np.sin(theta) * func_psi_u(
        deg_ground_basis, psi, theta, state, state_2, site
    )

    return psi_int


def TF_optimizer(
    site: int,
    state: int,
    K: np.ndarray,
    V: np.ndarray,
    grouped: bool = False,
) -> np.ndarray:

    state_2 = state**2

    U, _ = bz.thermofield_change_of_basis(K)
    I = np.eye(state)

    K_prim = oe.contract("pq,mw->pmqw", K, I, optimize="optimal")
    K_tilda_kron = U.T @ K_prim.reshape(state_2, state_2) @ U

    V_prim = oe.contract(
        "pqrs,mw,nv->pmqnrwsv",
        V.reshape(state, state, state, state),
        I,
        I,
        optimize="optimal",
    )

    U_kron = np.kron(U, U)

    V_tilda_kron = U_kron.T @ V_prim.reshape(state_2**2, state_2**2) @ U_kron

    H_TF, _, _ = hamiltonian_dense(
        state_2, site, 1, K_import=K_tilda_kron, V_import=V_tilda_kron, Import=True
    )

    print("Constduncted TF hamiltonian.")

    eig_val_TF, eig_vec_TF = np.linalg.eigh(H_TF)

    index_TF = np.argsort(eig_val_TF)

    deg_ground_basis = eig_vec_TF[:, index_TF[: state**site]]

    print("Constduncted degenerate ground basis.")

    C0 = np.full((state**site), 0, dtype=np.float64)

    res_1 = minimize(
        constr_ineq,
        x0=C0,
        args=(
            deg_ground_basis,
            state_2,
            site,
        ),
        method="SLSQP",
        constraints=[{"type": "eq", "fun": constr_eq}],
        options={"ftol": 1e-6, "maxiter": 1000},
    )

    print(res_1)

    C_opt = res_1.x
    psi_1 = deg_ground_basis @ C_opt
    D_1 = density_matrix_1(state**2, site, psi_1, 0)
    eig_val_opt, opt_grd_basis = np.linalg.eigh(D_1)
    index = np.argsort(-eig_val_opt)

    if not (np.allclose(C_opt @ C_opt, 1, atol=1e-10)):
        raise ValueError("Constants not normalized.")

    if not (
        np.allclose(
            f_max(C_opt, deg_ground_basis, state_2, site),
            np.trace(D_1 @ D_1).real,
            atol=1e-10,
        )
        and np.allclose(np.sum(eig_val_opt**2), np.trace(D_1 @ D_1).real, atol=1e-10)
    ):
        raise ValueError("D^2 is not matching.")

    res_2 = minimize(
        func_to_minmize,
        x0=0.00001,
        args=(
            psi_1,
            deg_ground_basis,
            state,
            state_2,
            site,
        ),
        method="SLSQP",
        options={"ftol": 1e-6, "maxiter": 1000},
    )

    print(res_2)

    psi_2 = min_output(
        res_2.x,
        psi_1,
        deg_ground_basis,
        state,
        state_2,
        site,
    )

    if not (
        np.allclose(
            res_2.fun,
            func_to_minmize(
                res_2.x,
                psi_1,
                deg_ground_basis,
                state,
                state_2,
                site,
            ),
        )
    ):
        raise ValueError("The optimization doesn't match the graph.")

    if not (
        res_2.fun.real
        > constr_ineq(
            C_opt,
            deg_ground_basis,
            state_2,
            site,
        )
    ):
        raise ValueError("The optimization step 2 is larger than step 1.")

    D_2 = density_matrix_1(state**2, site, psi_2, 0)

    eig_val_opt, opt_grd_basis = np.linalg.eigh(D_2)

    index_opt = np.argsort(-eig_val_opt)
    opt_grd_basis_orderd = opt_grd_basis[:, index_opt]

    return opt_grd_basis_orderd
