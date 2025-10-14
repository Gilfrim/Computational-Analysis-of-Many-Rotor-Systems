import numpy as np
import scipy as scp

from quant_rotor.models.dense.de_solver_func import new_solve_ivp
from quant_rotor.models.dense.support_ham import (
    basis_m_to_p_matrix_conversion,
    write_matrix_elements,
)
from quant_rotor.models.dense.t_amplitudes_sub_class import (
    QuantumSimulation,
    SimulationParams,
    TensorData,
)


def residual_double():
     return 0

def get_list_shape(lst):
    shape = []
    while isinstance(lst, list):
        shape.append(len(lst))
        if len(lst) == 0:
            break
        lst = lst[0]
    return tuple(shape)

def postprocess_rk45_integration_results(sol,t0_stored, state, site):

    # Copy the time value arrays
    time = sol.t.copy()

    # initialize the arrays to store the autocorrelation function
    true_evaluated_t0 = np.zeros_like(time, dtype=np.complex128)
    true_evaluated_T_ai = np.zeros_like(time, dtype=np.complex128)
    true_evaluated_two_max = np.zeros_like(time, dtype=np.complex128)

    # only extract the values which correspond to time steps in the solution
    # since we save C(t) for all integration steps, but only some are accepted

    t_dict = {t: (dT_0dB, T_ai, two_max) for (t, dT_0dB, T_ai, two_max) in t0_stored}

    for idx, t in enumerate(sol.t):
        dT_0dB, T_ai, two_max = t_dict[t]
        true_evaluated_t0[idx] = dT_0dB[0]
        true_evaluated_T_ai[idx] = T_ai
        true_evaluated_two_max[idx] = two_max

    return (time, true_evaluated_t0, true_evaluated_T_ai, true_evaluated_two_max)


def tdcc_differential_equation(t: float, comb_flat: np.ndarray, t0_stored, params: SimulationParams, tensors: TensorData, qs: QuantumSimulation) -> np.ndarray:
    """Set of coupled odes for a given time step numerically for the 1 electron hamiltonian_dict for the T_ai and T_0 equation for use
    in the scipy ode solver

    Parameters
    ----------
    t : float
        Some value of time
    T_ai_T_0_flat : np.array
        Flattened T_ai matrix concatenated with the T_0 value to be used in their respective ODE
    H : np.array
        1 electron hamiltonian_dict
    reference_state : int
        Number of electron to create the reference configuration used to the initial value problem of the coupled TDCC odes
        ex. reference_state = 2, has corresponding occupation number vector (1,1,0,0)
    thermofield : bool, optional
        Parameter for whether the hamiltonian_dict used is the thermofield hamiltonian_dict such that the correct partition of the hamiltonian_dict
        is used, by default False

    Returns
    -------
    np.array
        the flattened array containing the derivative of the T_ai and T_0 for a given time step, the 2 derivatives are concatenated
        to make a 1d array
    """
    site, a, p, i = params.site, params.a, params.p, params.i

    dTab_ijdB_sol, dTa_idB_sol, T_ai = (
        comb_flat[: -a - 1],
        comb_flat[-a - 1 : -1],
        comb_flat[-1],
    )
    dTab_ijdB = dTab_ijdB_sol.reshape(site, a, a, i, i)
    dTa_idB = dTa_idB_sol.reshape(a, i)

    tensors.t_a_i_tensor[0] = dTa_idB

    for site_1 in range(1, site):
        tensors.t_a_i_tensor[site_1] = tensors.t_a_i_tensor[0]
        tensors.t_ab_ij_tensor[0, site_1] = dTab_ijdB[site_1]
        for site_2 in range(1, site):
            tensors.t_ab_ij_tensor[site_2, (site_1 + site_2) % site] = tensors.t_ab_ij_tensor[0, site_1]

    two_max = tensors.t_ab_ij_tensor.flat[np.argmax(np.abs(tensors.t_ab_ij_tensor))]

    energy = 0

    for site_x in range(site):
        energy += np.einsum("ip, pi->", qs.h_term(i, p), qs.B_term(i, site_x)) #* 0.5

        for site_y in range(site_x + 1, site_x + site):
            # noinspection SpellCheckingInspection
            energy += np.einsum("ijab, abij->", qs.v_term(i, i, a, a, site_x, site_y % site), qs.t_term(site_x, site_y % site)) * 0.5
            # noinspection SpellCheckingInspection
            energy += np.einsum("ijpq, pi, qj->", qs.v_term(i, i, p, p, site_x, site_y % site), qs.B_term(i, site_x), qs.B_term(i, site_y % site)) * 0.5

    single = np.zeros((a,i), dtype = complex)
    double = np.zeros((site, a, a, i, i), dtype = complex)

    single = qs.residual_single(0)
    for y_site in range(1, site):
        double[y_site] = qs.residual_double_total(0, y_site)

    dTa_idB = (-1*(single))
    dTab_ijdB= (-1*(double))
    dT_0dB = [-1*(energy)]

    dTa_idB = dTa_idB.flatten()
    dTab_ijdB = dTab_ijdB.flatten()
    comb_flat = np.concatenate([dTab_ijdB, dTa_idB, dT_0dB])
    t0_stored.append((t, dT_0dB, T_ai, two_max))
    return (comb_flat)

def integration_scheme(site: int, state: int, g: float, t_init=0., t_final=10., nof_points=10000, K_import: np.ndarray=[], V_import: np.ndarray=[], import_K_V_TF = False, import_K_V_NO = False) -> tuple:
    """"""

    p = state
    i = 1
    a = p - i

    if import_K_V_TF:
        h_full = K_import
        v_full = V_import
    else:
        # Load .npy matrices directly from the package
        K, V = write_matrix_elements((state - 1) // 2)

        V_tensor = V.reshape(p, p, p, p)  # Adjust if needed

        h_full = basis_m_to_p_matrix_conversion(K, state)
        v_full = basis_m_to_p_matrix_conversion(V_tensor, state)

        v_full = v_full * g

    t_a_i_tensor = np.full((site, a, i), 0, dtype=complex)
    t_ab_ij_tensor = np.full((site, site, a, a, i, i), 0, dtype=complex)

    # eigenvalues from h for update
    epsilon = np.diag(h_full)

    params = SimulationParams(
    a=a,
    i=i,
    p=p,  # These can be the same as `a + i` or chosen independently
    site=site,
    state=state,
    i_method=3,
    gap=False,
    gap_site=3,
    epsilon=epsilon,
    periodic=True
    )

    tensors = TensorData(
    t_a_i_tensor=t_a_i_tensor,
    t_ab_ij_tensor=t_ab_ij_tensor,
    h_full=h_full,
    v_full=v_full
    )

    qs = QuantumSimulation(params, tensors)

    # Initialize T_0 (reference amplitude) as complex zero
    t_0 = complex(0)
    # Initialize T_ai amplitudes as zeros
    single = np.zeros((a,i), dtype = complex)
    double = np.zeros((site, a, a, i, i), dtype = complex)

    # Concatenate flattened T_ai and T_0 into a single array for the ODE solver
    init_amps = np.concatenate((double.flatten(), single.flatten(), np.array([t_0])),)

    step_size = (t_final - t_init) / nof_points

    # prepare the initial y_tensor
    t0_stored = [(0, 0, 0, 0)]  # time, value

    # Arguments to pass to the ODE function
    arguments = (t0_stored, params, tensors, qs)

    # specify the precision of the integrator so that the output for the test models is numerically identical
    relative_tolerance = 1e-9
    absolute_tolerance = 1e-10

    # ------------------------------------------------------------------------
    # call the integrator
    # ------------------------------------------------------------------------
    integration_function = tdcc_differential_equation

    sol = new_solve_ivp(
        fun=integration_function,  # the function we are integrating
        # method="RK45",  # the integration method we are using
        # method="RK23",  # the integration method we are using
        method='DOP853',
        first_step=step_size,  # fix the initial step size
        t_span=(
            t_init,  # initial time
            t_final,  # boundary time, integration end point
        ),
        y0=init_amps,  # initial state - shape (n, )
        args=arguments,  # extra args to pass to `rk45_solve_ivp_integration_function`
        max_step= 0.1,  # maximum allowed step size
        rtol=relative_tolerance,  # relative tolerance
        atol=absolute_tolerance,  # absolute tolerance
        store_y_values=False,  # do not store the y values over the integration
        t_eval=None,  # store all the time values that we integrated over
        dense_output=False,  # extra debug information
        # we do not need to vectorize
        # this means to process multiple time steps inside the function `rk45_solve_ivp_integration_function`
        # it would be useful for a method which does some kind of block stepping
        vectorized=False,
    )
    # ------------------------------------------------------------------------
    # now we extract the relevant information from the integrator object `sol`
    # ------------------------------------------------------------------------

    time, T_0, t_0_sol, two_max = postprocess_rk45_integration_results(sol,t0_stored, state, site)

    return(time, T_0, t_0_sol, two_max)
