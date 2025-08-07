import numpy as np
import scipy.sparse as sp
from importlib.resources import files
import opt_einsum as oe
from quant_rotor.models.dense.support_ham import write_matrix_elements, basis_m_to_p_matrix_conversion
from quant_rotor.models.dense.t_amplitudes_sub_class_new import QuantumSimulation, TensorData, SimulationParams, PrecalcalculatedTerms

#printout settings for large matrices
np.set_printoptions(suppress = True, linewidth = 1500, threshold = 10000, precision = 12)

def t_periodic(
    site: int,
    state: int,
    g: float,
    fast: bool, 
    i_method: int = 3,
    threshold: float=1e-8,
    gap: bool = False,
    gap_site: int=3,
    HF: bool=False,
    start_point: str="sin",
    low_state: int=1,
    t_a_i_tensor_initial: np.ndarray=0,
    t_ab_ij_tensor_initial: np.ndarray=0
):
    """
    Create SimulationParams from raw input arguments.
    Performs logic for a, i, p and validates start_point.
    """
    #state variables
    #could just use p, i, a
    #makes checking einsums and such a bit easier
    p = state
    i = low_state
    a = p - i
    periodic = True

    # Load .npy matrices directly from the package
    K, V = write_matrix_elements((state-1)//2)

    V = V + V.T - np.diag(np.diag(V))
    V_tensor = V.reshape(p, p, p, p)  # Adjust if needed

    h_full = basis_m_to_p_matrix_conversion(K, state).astype(np.float64)
    v_full = basis_m_to_p_matrix_conversion(V_tensor, state).astype(np.float64)

    v_full = v_full * g

    if np.isscalar(t_a_i_tensor_initial) and np.isscalar(t_ab_ij_tensor_initial):
        t_a_i_tensor = np.full((a), t_a_i_tensor_initial, dtype=np.float64)
        t_ab_ij_tensor = np.full((site, a, a), t_ab_ij_tensor_initial, dtype=np.float64)
    else:
        t_a_i_tensor = t_a_i_tensor_initial
        t_ab_ij_tensor = t_ab_ij_tensor_initial

    #eigenvalues from h for update
    epsilon = np.diag(h_full)

    params = SimulationParams(
    a=a,
    i=i,
    p=p,  # These can be the same as `a + i` or chosen independently
    site=site,
    state=state,
    i_method=i_method,
    gap=gap,
    gap_site=gap_site,
    epsilon=epsilon,
    periodic=periodic,
    fast=fast
    )

    tensors = TensorData(
    t_a_i_tensor=t_a_i_tensor,
    t_ab_ij_tensor=t_ab_ij_tensor,
    h_full=h_full,
    v_full=v_full
    )

    terms = PrecalcalculatedTerms()

    qs = QuantumSimulation(params, tensors, terms)

    del K, V, h_full, v_full, t_a_i_tensor, t_ab_ij_tensor, V_tensor

    iteration = 0

    single = np.zeros((a), dtype=np.float64)
    double = np.zeros((site, a, a), dtype=np.float64)

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
    print("Done.")
    while True:

        terms.a_term=qs.A_term(a)
        terms.b_term=qs.B_term(i)
        terms.bb_term=oe.contract("q,s->qs", terms.b_term, terms.b_term).reshape(p**2)
        terms.aa_term=oe.contract("ap,bq->abpq", terms.a_term, terms.a_term).reshape(a**2, p**2)

        energy = 0

        single = qs.residual_single()
        for y_site in range(1, site):
            # single[y_site] = single[0]
            double[y_site] = qs.residual_double_total(y_site)
            # for x_site in range(1, site):
            #     double[x_site, (x_site + y_site) % site] = double[0, y_site]

        one_max = single.flat[np.argmax(np.abs(single))]
        two_max = double.flat[np.argmax(np.abs(double))]

        tensors.t_a_i_tensor -= qs.update_one(single)

        for site_1 in range(1, site):
            # tensors.t_a_i_tensor[site_1] = tensors.t_a_i_tensor[0]
            tensors.t_ab_ij_tensor[site_1] -= qs.update_two(double[site_1])
            # for site_2 in range(1, site):
            #     tensors.t_ab_ij_tensor[site_2, (site_1 + site_2) % site] = tensors.t_ab_ij_tensor[0, site_1]

        #energy calculations
        # for site_x in range(site):
        #     energy += np.einsum("ip, pi->", terms.h_ip, terms.b_term) #* 0.5

        #     for site_y in range(site_x + 1, site_x + site):
        #         if abs(site_x - site_y) == 1 or abs(site_x - site_y) == (site - 1):
        #             V_iipp = terms.V_iipp
        #             V_iiaa = terms.V_iiaa
        #             # noinspection SpellCheckingInspection
        #             energy += np.einsum("ijab, abij->", V_iiaa, qs.t_term(site_x, site_y % site)) * 0.5
        #             # noinspection SpellCheckingInspection
        #             energy += np.einsum("ijpq, pi, qj->", V_iipp, terms.b_term, terms.b_term) * 0.5

        if np.all(abs(single) <= threshold) and np.all(abs(double) <= threshold):
            break

        #CHANGE BACK TO 10
        if abs(one_max) >= 100 or abs(two_max) >= 100:
            raise ValueError("Diverges.")

        iteration += 1
        # print(f"Iteration #: {iteration}")
        # print(f"Energy: {np.real(energy)}\n")

    return one_max, two_max, energy, tensors.t_a_i_tensor, tensors.t_ab_ij_tensor

# if __name__ == "__main__":

#     t_periodic(5, 5, 1, 0, 0, 1e-8, 0.5, 3,  False, 3, True, "sin")
#     print("Hello?")