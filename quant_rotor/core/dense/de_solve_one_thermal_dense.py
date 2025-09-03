from quant_rotor.models.dense.de_solver_func import new_solve_ivp
from quant_rotor.models.dense.t_amplitudes_sub_class_fast import QuantumSimulation, TensorData, SimulationParams, PrecalcalculatedTerms
import numpy as np 
import opt_einsum as oe
import thermofield_boltz_funcs as bz
import scipy.sparse as sp
from quant_rotor.models.sparse.support_ham import build_V_prime_in_p

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

        return(time,true_evaluated_t0, true_evaluated_T_ai, true_evaluated_two_max)
    
      
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

    dTab_ijdB_sol, dTa_idB_sol, T_ai = comb_flat[:-a-1], comb_flat[-a-1:-1], comb_flat[-1] 
    dTab_ijdB = dTab_ijdB_sol.reshape(site, a, a)
    dTa_idB = dTa_idB_sol.reshape(a)

    qs.tensors.t_a_i_tensor = dTa_idB

    for site_1 in range(1, site):
        qs.tensors.t_ab_ij_tensor[site_1] = dTab_ijdB[site_1]

    two_max = qs.tensors.t_ab_ij_tensor.flat[np.argmax(np.abs(qs.tensors.t_ab_ij_tensor))]

    qs.terms.a_term=qs.A_term(a)
    qs.terms.b_term=qs.B_term(i)
    qs.terms.bb_term=oe.contract("q,s->qs", qs.terms.b_term, qs.terms.b_term).reshape(p**2)
    qs.terms.aa_term=oe.contract("ap,bq->abpq", qs.terms.a_term, qs.terms.a_term).reshape(a**2, p**2)

    energy = 0

    for site_x in range(site):
        energy += qs.terms.h_ip @ qs.terms.b_term

        for site_y in range(site_x + 1, site_x + site):
            if abs(site_x - site_y) == 1 or abs(site_x - site_y) == (site - 1):
                V_iipp = qs.terms.V_iipp
                V_iiaa = qs.terms.V_iiaa
                T_xy = qs.t_term(site_x, site_y)

                # noinspection SpellCheckingInspection
                energy +=  np.sum(V_iiaa* (T_xy)) * 0.5
                # noinspection SpellCheckingInspection
                energy += (V_iipp @ qs.terms.b_term @ qs.terms.b_term) * 0.5

    single = np.zeros((a), dtype = complex)
    double = np.zeros((site, a, a), dtype = complex)

    single = qs.residual_single()
    for y_site in range(1, site):
        double[y_site] = qs.residual_double_total(y_site)

    dTab_ijdB = (-1*(double))
    dTa_idB = (-1*(single))
    dT_0dB = [-1*(energy)]

    dTa_idB = dTa_idB.flatten()
    dTab_ijdB = dTab_ijdB.flatten()
    comb_flat = np.concatenate([dTab_ijdB, dTa_idB, dT_0dB])
    t0_stored.append((t, dT_0dB, T_ai, two_max))
    return (comb_flat)

def integration_scheme(site: int, state: int, g: float, t_init=0., t_final=10., nof_points=10000, K_import: np.ndarray=[], V_import: np.ndarray=[], import_K_V_TF = False, import_K_V_NO = False) -> tuple:
    """"""

    # Load .npy matrices directly from the package
    K, V = build_V_prime_in_p(state)
    I = np.eye(state)

    U, _ = bz.thermofield_change_of_basis(I)

    U_sparse = sp.csr_matrix(U)

    h_full = (U_sparse.T @ K @ U_sparse).toarray()

    v_full = oe.contract('Mi,Wj,ijab,aN,bV->MWNV', U, U, V.toarray().reshape(state**2, state**2, state**2, state**2), U, U, optimize='optimal')
    v_full = v_full * g

    state = state**2
    p = state
    i = 1
    a = p - i

    t_a_i_tensor = np.full((a), 0, dtype=complex)
    t_ab_ij_tensor = np.full((site, a, a), 0, dtype=complex)

    #eigenvalues from h for update
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

    terms = PrecalcalculatedTerms()
    qs = QuantumSimulation(params, tensors, terms)

    del K, V, h_full, v_full, t_a_i_tensor, t_ab_ij_tensor
    
    # Initialize T_0 (reference amplitude) as complex zero
    t_0 = complex(0)
    # Initialize T_ai amplitudes as zeros
    single = np.zeros((a), dtype = complex)
    double = np.zeros((site, a, a), dtype = complex)
    
    terms.h_pp=qs.h_term(p, p)
    terms.h_pa=qs.h_term(p, a)
    terms.h_ip=qs.h_term(i, p).reshape(p)
    terms.V_pppp=qs.v_term(p, p, p, p, 0, 1).reshape(p**2, p**2)
    terms.V_ppaa=qs.v_term(p, p, a, a, 0, 1).reshape(p**2, a**2)
    terms.V_iipp=qs.v_term(i, i, p, p, 0, 1).reshape(p, p)
    terms.V_iiaa=qs.v_term(i, i, a, a, 0, 1).reshape(a, a)
    terms.V_piaa=qs.v_term(p, i, a, a, 0, 1).reshape(p, a**2)
    terms.V_pipp=qs.v_term(p, i, p, p, 0, 1).reshape(p, p**2)
    terms.V_ipap=qs.v_term(i, p, a, p, 0, 1).reshape(p, a, p)
    terms.V_piap=qs.v_term(p, i, a, p, 0, 1).reshape(p, a, p)

    # Concatenate flattened T_ai and T_0 into a single array for the ODE solver
    init_amps = np.concatenate((double.flatten(), single.flatten(), np.array([t_0])))

    step_size = (t_final - t_init) / nof_points

    # prepare the initial y_tensor
    t0_stored = [(0, 0, 0, 0)]  # time, value 

    # Arguments to pass to the ODE function
    arguments = (t0_stored, params, tensors, qs)
    
    # specify the precision of the integrator so that the output for the test models is numerically identical
    relative_tolerance = 1e-9
    absolute_tolerance = 1e-10

    print("Done.")

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