from typing import Tuple, List, Dict, Union, Callable 
from quant_rotor.models.temporary_trial_solver import new_solve_ivp
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


def postprocess_rk45_integration_results(sol,t0_stored):
        
        # Copy the time value arrays
        time = sol.t.copy()

        # initialize the arrays to store the autocorrelation function
        true_evaluated_t0 = np.zeros_like(time, dtype=complex)
         
        # only extract the values which correspond to time steps in the solution
        # since we save C(t) for all integration steps, but only some are accepted
        t_dict = {t[0]: t[1] for t in t0_stored}
        for idx, t in enumerate(sol.t):
            true_evaluated_t0[idx] = t_dict[t]

        return(time,true_evaluated_t0)
    
      
def tdcc_differential_equation(t:float,T_ai_T_0_flat:np.array,hamiltonian_dict:np.array,t0_stored) -> np.array:
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

    h_o_o = hamiltonian_dict['h_occ_occ']
    h_v_v = hamiltonian_dict['h_vrt_vrt']
    h_v_o = hamiltonian_dict['h_vrt_occ']
    h_o_v = hamiltonian_dict['h_occ_vrt']

    n, m = np.shape(h_v_o)
    T_ai_flat, T_0 = T_ai_T_0_flat[:n*m],T_ai_T_0_flat[n*m] 
    T_ai = (T_ai_flat.reshape(np.shape(h_v_o)))

    dTaidt = (-1j*(h_v_o 
                    -np.einsum('ji,aj->ai',h_o_o,T_ai) 
                    +np.einsum('ab,bi->ai',h_v_v,T_ai)
                    -np.einsum('kb,bi,ak-> ai',h_o_v,T_ai,T_ai)))
    
    dT_0dt = [-1j*(np.trace(h_o_o)+  np.einsum('ia,ai->',h_o_v,T_ai))]
    
    
    dTaidt = dTaidt.flatten()
    dT_aiT_0_flat = np.concatenate([dTaidt,dT_0dt])
    t0_stored.append((t,T_0))

    return (dT_aiT_0_flat)

def integration_scheme(hamiltonian_dict:dict, t_init=0., t_final=10., nof_points=10000 ) -> Tuple:
    """"""     
 
    # Get the shape of the T_ai amplitude matrix from the Hamiltonian
    h_v_o_shape = hamiltonian_dict['h_vrt_occ'].shape
    
    # Initialize T_0 (reference amplitude) as complex zero
    t_0 = complex(0)
    # Initialize T_ai amplitudes as zeros
    t_ai = np.zeros(h_v_o_shape, dtype=complex)
    
    # Concatenate flattened T_ai and T_0 into a single array for the ODE solver
    init_amps = np.concatenate((t_ai.flatten(), np.array([t_0])),)
    
 
    step_size = (t_final - t_init) / nof_points

    # prepare the initial y_tensor
    t0_stored = [(0,0)]  # time, value 

    # Arguments to pass to the ODE function
    arguments = (hamiltonian_dict,t0_stored,[])
    
    # specify the precision of the integrator so that the output for the test models is numerically identical
    relative_tolerance = 1e-10
    absolute_tolerance = 1e-12

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
    
    time,T_0 = postprocess_rk45_integration_results(sol,t0_stored)
    
    return(time,T_0)

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
time,t_0 = integration_scheme(hamiltonian_dict,t_init=0,t_final=100,nof_points=1000)

breakpoint()

# Demonstrate propagation works by plotting autocorrelation function
plt.plot(time,(np.exp(t_0)))
plt.show()