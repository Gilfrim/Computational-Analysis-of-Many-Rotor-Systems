from typing import Tuple, List, Dict, Union, Callable 
from quant_rotor.models.dense.de_solver_func import new_solve_ivp
from quant_rotor.models.dense.t_amplitudes_sub_class import QuantumSimulation, TensorData, SimulationParams
from quant_rotor.models.dense.support_ham import write_matrix_elements, basis_m_to_p_matrix_conversion
import numpy as np 
import scipy as scp
import matplotlib.pyplot as plt

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

def postprocess_rk45_integration_results(sol,t0_stored, states, sites):
        
        # Copy the time value arrays
        time = sol.t.copy()

        # initialize the arrays to store the autocorrelation function
        true_evaluated_t0 = np.zeros_like(time, dtype=np.complex128)
        single_max = np.zeros_like(time, dtype=np.complex128)
        double_max = np.zeros_like(time, dtype=np.complex128)
        t_1_arr = np.zeros((len(time), states-1), dtype=np.complex128)
        t_2_arr = np.zeros((len(time), sites*(states-1)**2), dtype=np.complex128)
         
        # only extract the values which correspond to time steps in the solution
        # since we save C(t) for all integration steps, but only some are accepted
        
        t_dict = {t: (dT_0dt, one_max, two_max, t_1, t_2) for (t, dT_0dt, one_max, two_max, t_1, t_2) in t0_stored}
        
        for idx, t in enumerate(sol.t):
            dT_0dt, one_max, two_max, t_1, t_2 = t_dict[t]
            true_evaluated_t0[idx] = dT_0dt[0]
            single_max[idx] = one_max
            double_max[idx] = two_max
            t_1_arr[idx, :] = t_1
            t_2_arr[idx, :] = t_2


        return(time,true_evaluated_t0, single_max, double_max, t_1_arr, t_2_arr)
    
      
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
    sites, a, p, i = params.sites, params.a, params.p, params.i
    

    dTab_ijdt_sol, dTa_idt_sol, T_ai = comb_flat[:-a-1], comb_flat[-a-1:-1], comb_flat[-1] 
    dTab_ijdt = dTab_ijdt_sol.reshape(sites, a, a, i, i)
    dTa_idt = dTa_idt_sol.reshape(a, i)

    tensors.t_a_i_tensor[0] = dTa_idt

    for site_1 in range(1, sites):
        tensors.t_a_i_tensor[site_1] = tensors.t_a_i_tensor[0]
        tensors.t_ab_ij_tensor[0, site_1] = dTab_ijdt[site_1]
        for site_2 in range(1, sites):
            tensors.t_ab_ij_tensor[site_2, (site_1 + site_2) % sites] = tensors.t_ab_ij_tensor[0, site_1]

    one_max = tensors.t_a_i_tensor.flat[np.argmax(np.abs(tensors.t_a_i_tensor))]
    two_max = tensors.t_ab_ij_tensor.flat[np.argmax(np.abs(tensors.t_ab_ij_tensor))]

    # print(f"Time: {t} 1 max: {one_max}")
    # print(f"2 max: {two_max}")

    energy = 0

    for site_x in range(sites):
        energy += np.einsum("ip, pi->", qs.h_term(i, p), qs.B_term(i, site_x)) * 0.5

        for site_y in range(site_x + 1, site_x + sites):
            # noinspection SpellCheckingInspection
            energy += np.einsum("ijab, abij->", qs.v_term(i, i, a, a, site_x, site_y % sites), qs.t_term(site_x, site_y % sites)) * 0.5
            # noinspection SpellCheckingInspection
            energy += np.einsum("ijpq, pi, qj->", qs.v_term(i, i, p, p, site_x, site_y % sites), qs.B_term(i, site_x), qs.B_term(i, site_y % sites)) * 0.5

    single = np.zeros((a,i), dtype = complex)
    double = np.zeros((sites, a, a, i, i), dtype = complex)

    single = qs.residual_single(0)
    for y_site in range(1, sites):
        double[y_site] = qs.residual_double_total(0, y_site)


    
    dTa_idt = (-1j*(single))

    dTab_ijdt= (-1j*(double))
    
    dT_0dt = [-1j*(energy)]

    dTa_idt = dTa_idt.flatten()
    dTab_ijdt = dTab_ijdt.flatten()
    comb_flat = np.concatenate([dTab_ijdt, dTa_idt,dT_0dt])
    t0_stored.append((t, dT_0dt, one_max, two_max, dTa_idt_sol, dTab_ijdt_sol))
    return (comb_flat)

def integration_scheme(sites: int, states: int, g: float, t_init=0., t_final=10., nof_points=10000, K_import: np.ndarray=[], V_import: np.ndarray=[], import_K_V = False) -> Tuple:
    """"""     


    p = states
    i = 1
    a = p - i

    # Load .npy matrices directly from the package
    K, V = write_matrix_elements((states-1)//2)

    V = V + V.T - np.diag(np.diag(V))
    V_tensor = V.reshape(p, p, p, p)  # Adjust if needed

    if import_K_V:
        h_full = K_import
        v_full = V_import.reshape(sites**states, sites**states, sites**states, sites**states)
    else:
        h_full = basis_m_to_p_matrix_conversion(K)
        v_full = basis_m_to_p_matrix_conversion(V_tensor)

        v_full = v_full * g

    print(v_full.shape)

    t_a_i_tensor = np.full((sites, a, i), 0, dtype=complex)
    t_ab_ij_tensor = np.full((sites, sites, a, a, i, i), 0, dtype=complex)

    #eigenvalues from h for update
    epsilon = np.diag(h_full)

    params = SimulationParams(
    a=a,
    i=i,
    p=p,  # These can be the same as `a + i` or chosen independently
    sites=sites,
    states=states,
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

    del K, V, h_full, v_full, t_a_i_tensor, t_ab_ij_tensor, V_tensor

    params.epsilon = np.diag(tensors.h_full)
    
    # Initialize T_0 (reference amplitude) as complex zero
    t_0 = complex(0)
    # Initialize T_ai amplitudes as zeros
    single = np.zeros((a,1), dtype = complex)
    double = np.zeros((sites, a, a, i, i), dtype = complex)
    
    # Concatenate flattened T_ai and T_0 into a single array for the ODE solver
    init_amps = np.concatenate((double.flatten(), single.flatten(), np.array([t_0])),)

    step_size = (t_final - t_init) / nof_points

    # prepare the initial y_tensor
    t0_stored = [(0, 0, 0, 0, single, double)]  # time, value 

    # Arguments to pass to the ODE function
    arguments = (t0_stored, params, tensors, qs)
    
    # specify the precision of the integrator so that the output for the test models is numerically identical
    relative_tolerance = 1e-5
    absolute_tolerance = 1e-7

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
    
    time, T_0, t1_max, t_2max, t_1, t_2 = postprocess_rk45_integration_results(sol,t0_stored, states, sites)
    
    return(time, T_0, t1_max, t_2max, t_1, t_2)