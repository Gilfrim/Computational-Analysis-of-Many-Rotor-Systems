from typing import Tuple, List, Dict, Union, Callable 
import numpy as np 
import scipy as scp
import matplotlib.pyplot as plt

def get_hamiltonian_dict(hamiltonian:np.array, n_occ:int) -> dict:
    """Partitions the 1 electron hamiltonian into virtual, occupied, and mixed blocks, can also use thermofield transformed hamiltonian  

    Parameters
    ----------
    H : np.array
        1 electron hamiltonian 
    n_occ : int
        number of occupied electrons

    Returns
    -------
    H_dict: dict
        Dictionary of the truncated hamiltonian 
    """   

    h_occ_occ = hamiltonian[:n_occ,:n_occ]  
    h_vrt_vrt = hamiltonian[n_occ:, n_occ:]
    h_vrt_occ = hamiltonian[n_occ:, :n_occ]
    h_occ_vrt = hamiltonian[:n_occ, n_occ:]

    h_dict ={'h_occ_occ':h_occ_occ, 'h_vrt_vrt':h_vrt_vrt, 'h_vrt_occ':h_vrt_occ, 'h_occ_vrt':h_occ_vrt}

    return(h_dict)

def tdcc_differential_equation(
    t: float,
    t_ai_t_0_flat: np.array,
    hamiltonian_dict: Dict
) -> np.array:
    """
    Computes the time derivative of the coupled cluster amplitudes (T_ai) and reference amplitude (T_0)
    for the time-dependent coupled cluster (TDCC) equations using the provided Hamiltonian.

    Parameters
    ----------
    t : float
        Current time value (not used explicitly in this function, but required for ODE solvers).
    t_ai_t_0_flat : np.array
        Flattened array containing both T_ai amplitudes and T_0 amplitude concatenated.
    hamiltonian_dict : Dict
        Dictionary containing the Hamiltonian matrix elements:
            - 'h_occ_occ': Occupied-occupied block
            - 'h_vrt_vrt': Virtual-virtual block
            - 'h_vrt_occ': Virtual-occupied block
            - 'h_occ_vrt': Occupied-virtual block

    Returns
    -------
    np.array
        Flattened array of the time derivatives of T_ai and T_0 amplitudes.
    """

    # Extract Hamiltonian matrix elements from the dictionary
    h_o_o = hamiltonian_dict['h_occ_occ']
    h_v_v = hamiltonian_dict['h_vrt_vrt']
    h_v_o = hamiltonian_dict['h_vrt_occ']
    h_o_v = hamiltonian_dict['h_occ_vrt']

    # Determine the shape of the T_ai amplitudes
    n, m = np.shape(h_v_o)
    # Separate T_ai and T_0 from the input array
    T_ai_flat, T_0 = t_ai_t_0_flat[:n*m], t_ai_t_0_flat[n*m] 
    T_ai = T_ai_flat.reshape(np.shape(h_v_o))

    # Compute the time derivative of T_ai using Einstein summation for efficient tensor contractions
    dTaidt = (-1j * (
        h_v_o 
        - np.einsum('ji,aj->ai', h_o_o, T_ai) 
        + np.einsum('ab,bi->ai', h_v_v, T_ai)
        - np.einsum('kb,bi,ak->ai', h_o_v, T_ai, T_ai)
    ))
    
    # Compute the time derivative of T_0 (reference amplitude)
    dT_0dt = [-1j * (np.trace(h_o_o) + np.einsum('ia,ai->', h_o_v, T_ai))]
    
    # Flatten dTaidt and concatenate with dT_0dt for ODE solver compatibility
    dTaidt = dTaidt.flatten()
    dT_aiT_0_flat = np.concatenate([dTaidt, dT_0dt])
    
    return dT_aiT_0_flat


def solve_ivp_wrapper_T_0(
    ode_func: str,
    hamiltonian_dict: np.array,
    final_time: float,
    step_size: float,
    method_of_integration: str,
) -> Tuple[np.array, np.array, np.array]:
    """
    Wrapper function for scipy's solve_ivp to solve the time-dependent coupled cluster equations.
    Initializes amplitudes, sets up integration parameters, and returns the time evolution of T_0 and T_ai.

    Parameters
    ----------
    ode_func : str
        Name of the ODE function to solve (should be compatible with scipy's solve_ivp).
    hamiltonian_dict : np.array
        Dictionary containing the Hamiltonian matrix elements.
    final_time : float
        Final time up to which the ODE is integrated.
    step_size : float
        Time step size for the integration.
    method_of_integration : str
        Integration method to use (e.g., 'RK45', 'BDF', etc.).

    Returns
    -------
    Tuple[np.array, np.array, np.array]
        - time_values: Array of time points at which the solution was evaluated.
        - t_0_values: List of T_0 amplitudes at each time point.
        - t_ai_matrix_values: List of T_ai matrices at each time point.
    """
    
    # Get the shape of the T_ai amplitude matrix from the Hamiltonian
    h_v_o_shape = hamiltonian_dict['h_vrt_occ'].shape
    
    # Initialize T_0 (reference amplitude) as complex zero
    t_0 = complex(0)
    # Initialize T_ai amplitudes as zeros
    t_ai = np.zeros(h_v_o_shape, dtype=complex)
    
    # Concatenate flattened T_ai and T_0 into a single array for the ODE solver
    init_amps = np.concatenate((t_ai.flatten(), np.array([t_0])),)
    
    # Arguments to pass to the ODE function
    arguments = hamiltonian_dict
    
    # Solve the ODE using scipy's solve_ivp
    sol = scp.integrate.solve_ivp(
        ode_func, 
        (0, final_time + step_size), 
        init_amps, 
        t_eval=np.arange(0, final_time + step_size, step_size), 
        args=(arguments,),
        method=method_of_integration,
        rtol=1e-10,
        atol=1e-12
    )
    
    # Extract time points and solution values
    time_values = sol.t
    y_solutions = sol.y.T
    
    # Number of elements in T_ai
    num_elems = np.prod(h_v_o_shape)
    
    # Extract T_0 values at each time point
    t_0_values = [y_solutions[i][num_elems] for i in range(len(time_values))]
    # Extract T_ai matrices at each time point
    t_ai_matrix_values = [
        (sol.y.T[i][:num_elems]).reshape(h_v_o_shape) 
        for i in range(len(time_values))
    ]
    
    return (time_values, t_0_values, t_ai_matrix_values)

# Test hamiltonian 
hamiltonian  = np.array([[-1, 0.2, 0.3, 0.4],
                        [ 0.2, -3, 0.7, 0.5], 
                        [ 0.3,  0.7, 3, 0.6],
                        [ 0.4,  0.5, 0.6, 2]])

# Number of electrons 
n_el = 2 

# Partition hamiltonian (specific to my case)
hamiltonian_dict = get_hamiltonian_dict(hamiltonian,n_el)

# Call function to propagate 
time,t_0,t_ai = solve_ivp_wrapper_T_0(tdcc_differential_equation,hamiltonian_dict,100,0.01,'DOP853')

# Demonstrate propagation works by plotting autocorrelation function
plt.plot(time,np.exp(t_0))
plt.show()