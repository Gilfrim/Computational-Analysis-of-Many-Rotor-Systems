import numpy as np
import sys
import scipy.sparse as sp
import opt_einsum as oe
from quant_rotor.models.dense.support_ham import write_matrix_elements, basis_m_to_p_matrix_conversion
import quant_rotor.models.dense.t_amplitudes_sub_class_fast as new
import quant_rotor.models.dense.t_amplitudes_sub_class as old

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

    h_full = basis_m_to_p_matrix_conversion(K, state)
    v_full = basis_m_to_p_matrix_conversion(V_tensor, state)

    v_full = v_full * g

    if np.isscalar(t_a_i_tensor_initial) and np.isscalar(t_ab_ij_tensor_initial):
        t_a_i_tensor_new = np.full((a), t_a_i_tensor_initial, dtype=complex)
        t_ab_ij_tensor_new = np.full((site, a, a), t_ab_ij_tensor_initial, dtype=complex)

        t_a_i_tensor_old = np.full((site, a, i), 0, dtype=complex)
        t_ab_ij_tensor_old = np.full((site, site, a, a, i, i), 0, dtype=complex)
    else:
        t_a_i_tensor = t_a_i_tensor_initial
        t_ab_ij_tensor = t_ab_ij_tensor_initial

    #eigenvalues from h for update
    epsilon = np.diag(h_full)

    print(epsilon)

    params_old = old.SimulationParams(
    a=a,
    i=i,
    p=p,
    site=site,
    state=state,
    i_method=i_method,
    gap=gap,
    gap_site=gap_site,
    epsilon=epsilon,
    periodic=periodic,
    )

    tensors_old = old.TensorData(
    t_a_i_tensor=t_a_i_tensor_old,
    t_ab_ij_tensor=t_ab_ij_tensor_old,
    h_full=h_full,
    v_full=v_full
    )

    params_new = new.SimulationParams(
    a=a,
    i=i,
    p=p,
    site=site,
    state=state,
    i_method=i_method,
    gap=gap,
    gap_site=gap_site,
    epsilon=epsilon,
    periodic=periodic,
    fast=fast
    )

    tensors_new = new.TensorData(
    t_a_i_tensor=t_a_i_tensor_new,
    t_ab_ij_tensor=t_ab_ij_tensor_new,
    h_full=h_full,
    v_full=v_full
    )

    terms = new.PrecalcalculatedTerms()

    qs_new = new.QuantumSimulation(params_new, tensors_new, terms)
    qs_old = old.QuantumSimulation(params_old, tensors_old)

    del K, V, h_full, v_full, t_a_i_tensor_old, t_ab_ij_tensor_old,  V_tensor, t_a_i_tensor_new, t_ab_ij_tensor_new

    iteration = 0

    single_new = np.zeros((a), dtype=complex)
    double_new = np.zeros((site, a, a), dtype=complex)

    single_old = np.zeros((site, a, i), dtype = complex)
    double_old = np.zeros((site, site, a, a, i ,i), dtype = complex)

    terms.h_pp=qs_new.h_term(p, p)
    terms.h_pa=qs_new.h_term(p, a)
    terms.h_ip=qs_new.h_term(i, p).reshape(p)
    terms.V_pppp=qs_new.v_term(p, p, p, p, 0, 1).reshape(p**2, p**2)
    terms.V_ppaa=qs_new.v_term(p, p, a, a, 0, 1).reshape(p**2, a**2)
    terms.V_iipp=qs_new.v_term(i, i, p, p, 0, 1).reshape(p, p)
    terms.V_iiaa=qs_new.v_term(i, i, a, a, 0, 1).reshape(a, a)
    terms.V_piaa=qs_new.v_term(p, i, a, a, 0, 1).reshape(p, a**2)
    terms.V_pipp=qs_new.v_term(p, i, p, p, 0, 1).reshape(p, p**2)
    terms.V_ipap=qs_new.v_term(i, p, a, p, 0, 1).reshape(p, a, p)
    terms.V_piap=qs_new.v_term(p, i, a, p, 0, 1).reshape(p, a, p)
    print("Done.")

    while True:

        terms.a_term=qs_new.A_term(a)
        terms.b_term=qs_new.B_term(i)
        terms.bb_term=oe.contract("q,s->qs", terms.b_term, terms.b_term).reshape(p**2)
        terms.aa_term=oe.contract("ap,bq->abpq", terms.a_term, terms.a_term).reshape(a**2, p**2)

        energy = 0

        single_new = qs_new.residual_single()
        for y_site in range(1, site):
            # single[y_site] = single[0]
            double_new[y_site] = qs_new.residual_double_total(y_site)
            # for x_site in range(1, site):
            #     double[x_site, (x_site + y_site) % site] = double[0, y_site]

        single_old[0] = qs_old.residual_single(0)
        for y_site in range(1, site):
            single_old[y_site] = single_old[0]
            double_old[0, y_site] = qs_old.residual_double_total(0, y_site)
            for x_site in range(1, site):
                double_old[x_site, (x_site + y_site) % site] = double_old[0, y_site]

        one_max_new = single_new.flat[np.argmax(np.abs(single_new))]
        two_max_new = double_new.flat[np.argmax(np.abs(double_new))]

        one_max_old = single_old.flat[np.argmax(np.abs(single_old))]
        two_max_old = double_old.flat[np.argmax(np.abs(double_old))]



        if np.array_equal(one_max_old, one_max_old):
            print(f"One Max: {np.array_equal(one_max_old, one_max_old)}")
        else:
            print(f"Delta One Max: {np.abs(one_max_old - one_max_old)}")

        if np.array_equal(two_max_old, two_max_new):
            print(f"Two Max: {np.array_equal(two_max_old, two_max_new)}", "\n")
        else:
            print(f"Delta Two Max: {np.abs(two_max_old - two_max_new)}", "\n")

        # if np.allclose(qs_old.update_one(single_old[0].reshape(state-1)), qs_new.update_one(single_new), 1e-15):
        #     print(f"Singles: {np.allclose(qs_old.update_one(single_old[0].reshape(state-1)), qs_new.update_one(single_new), 1e-10)}", "\n")
        # else:
        #     diff = np.abs(qs_new.update_one(single_new) - qs_old.update_one(single_old[0].reshape(state-1)))
        #     max_diff = np.max(diff)
        #     print("Singles max absolute difference:", max_diff, "\n")

        if np.allclose(single_old[0].reshape(state-1), single_new, 1e-15):
            print(f"Singles: {np.allclose(single_old[0].reshape(state-1), single_new, 1e-15)}", "\n")
        else:
            diff = np.abs(single_new - single_old[0].reshape(state-1))
            max_diff = np.max(diff)
            print("Singles max absolute difference:", max_diff, "\n")

        tensors_new.t_a_i_tensor -= qs_new.update_one(single_new)

        for site_1 in range(1, site):
            # tensors.t_a_i_tensor[site_1] = tensors.t_a_i_tensor[0]
            tensors_new.t_ab_ij_tensor[site_1] -= qs_new.update_two(double_new[site_1])
            # for site_2 in range(1, site):
            #     tensors.t_ab_ij_tensor[site_2, (site_1 + site_2) % site] = tensors.t_ab_ij_tensor[0, site_1]

        tensors_old.t_a_i_tensor[0] -= qs_old.update_one(single_old[0])

        for site_1 in range(1, site):
            tensors_old.t_a_i_tensor[site_1] = tensors_old.t_a_i_tensor[0]
            tensors_old.t_ab_ij_tensor[0, site_1] -= qs_old.update_two(double_old[0, site_1])
            for site_2 in range(1, site):
                tensors_old.t_ab_ij_tensor[site_2, (site_1 + site_2) % site] = tensors_old.t_ab_ij_tensor[0, site_1]



        if np.allclose(tensors_old.t_a_i_tensor[0].reshape(state-1), tensors_new.t_a_i_tensor, 1e-20):
            print(f"t_1: {np.allclose(tensors_old.t_a_i_tensor[0].reshape(state-1), tensors_new.t_a_i_tensor, 1e-20)}")
        else:
            diff = np.abs(tensors_new.t_a_i_tensor - tensors_old.t_a_i_tensor[0].reshape(state-1))
            max_diff = np.max(diff)
            print("t_1 max absolute difference:", max_diff)

        if np.allclose(tensors_old.t_ab_ij_tensor[0].reshape(site, state-1, state-1), tensors_new.t_ab_ij_tensor, 1e-20):
            print(f"t_2: {np.allclose(tensors_old.t_ab_ij_tensor[0].reshape(site, state-1, state-1), tensors_new.t_ab_ij_tensor, 1e-20)}", "\n")
        else:
            diff = np.abs(tensors_new.t_ab_ij_tensor - tensors_old.t_ab_ij_tensor[0].reshape(site, state-1, state-1))
            max_diff = np.max(diff)
            print("t_2 max absolute difference:", max_diff, "\n\n")



        if np.all(abs(single_old) <= threshold) and np.all(abs(double_old) <= threshold):
            break

        #CHANGE BACK TO 10
        if abs(one_max_old) >= 100 or abs(two_max_old) >= 100:
            raise ValueError("Diverges.")

        iteration += 1
        # sys.stdout.write("\033[F" * 4)  # ANSI escape: move cursor up
        # sys.stdout.flush()
    return one_max_old, two_max_old, energy, tensors_old.t_a_i_tensor[0], tensors_old.t_ab_ij_tensor[0]

if __name__ == "__main__":

    state = 11
    site = 5
    g = 0.1

    t_periodic(site, state, g, True)