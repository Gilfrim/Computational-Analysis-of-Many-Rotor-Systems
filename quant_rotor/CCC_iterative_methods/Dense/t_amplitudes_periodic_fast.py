import numpy as np

from quant_rotor.CCC_iterative_methods.Dense.t_amplitudes_sub_class_fast import (
    PrecalcalculatedTerms,
    QuantumSimulation,
    SimulationParams,
    TensorData,
)

# printout settings for large matrices
np.set_printoptions(suppress=True, linewidth=1500, threshold=10000, precision=12)


def t_periodic(
    site: int,
    state: int,
    h_full: np.ndarray,
    v_full_xy: np.ndarray,
    v_full_yx: np.ndarray,
    threshold: float = 1e-10,
    gap: bool = False,
    gap_site: int = 3,
    low_state: int = 1,
    Import_t: bool = False,
    t_1_import: np.ndarray = [],
    t_2_import: np.ndarray = [],
    periodic: bool = True,
    one_cicle: bool = False,
):
    """
    Create SimulationParams from raw input arguments.
    Performs logic for a, i, p and validates start_point.
    """
    # state variables
    # could just use p, i, a
    # makes checking einsums and such a bit easier
    p = state
    i = low_state
    a = p - i

    if Import_t:
        t_a_i_tensor = t_1_import
        t_ab_ij_tensor = t_2_import
    else:
        t_a_i_tensor = np.full((site, a), 0.0, dtype=complex)
        t_ab_ij_tensor = np.full((site, site, a, a), 0.0, dtype=complex)

    # eigenvalues from h for update
    epsilon = np.diag(h_full)

    params = SimulationParams(
        a=a,
        i=i,
        p=p,  # These can be the same as `a + i` or chosen independently
        site=site,
        state=state,
        gap=gap,
        gap_site=gap_site,
        epsilon=epsilon,
        periodic=periodic,
    )

    tensors = TensorData(
        t_a_i_tensor=t_a_i_tensor,
        t_ab_ij_tensor=t_ab_ij_tensor,
        h_full=h_full,
        v_full_xy=v_full_xy,
        v_full_yx=v_full_yx,
    )

    terms = PrecalcalculatedTerms()

    qs = QuantumSimulation(params, tensors, terms)

    iteration = 0

    single = np.zeros((site, a), dtype=complex)
    double = np.zeros((site, site, a, a), dtype=complex)

    terms.h_pp = qs.h_term(p, p)
    terms.h_pa = qs.h_term(p, a)
    terms.h_ip = qs.h_term(i, p).reshape(p)

    while True:

        # terms.a_term = qs.A_term(a)
        # terms.b_term = qs.B_term(i)
        # terms.bb_term = np.einsum("q,s->qs", terms.b_term, terms.b_term).reshape(p**2)
        # terms.aa_term = np.einsum("ap,bq->abpq", terms.a_term, terms.a_term).reshape(
        #     a**2, p**2
        # )

        energy = 0

        for x in range(site):
            single[x] = qs.residual_single(x)
            for y in range(site):
                if x < y:
                    double[x, y] = qs.residual_double_total(x, y)
                    double[y, x] = double[x, y].T

        if one_cicle:
            return t_a_i_tensor, t_ab_ij_tensor, single, double

        one_max = single.flat[np.argmax(np.abs(single))]
        two_max = double.flat[np.argmax(np.abs(double))]

        for x in range(site):
            tensors.t_a_i_tensor[x] -= qs.update_one(single[x])

            for y in range(site):
                if x < y:
                    tensors.t_ab_ij_tensor[x, y] -= qs.update_two(double[x, y])
                    tensors.t_ab_ij_tensor[y, x] -= qs.update_two(double[y, x])

        iteration += 1

        print(one_max, two_max)

        if np.all(abs(single) <= threshold) and np.all(abs(double) <= threshold):
            break

        # CHANGE BACK TO 10
        if abs(one_max) >= 100 or abs(two_max) >= 100:
            raise ValueError("Diverges.")

    if periodic:
        energy += terms.h_ip @ qs.B_term(0)

        V_iipp_01 = qs.v_term(i, i, p, p, 0, 1).reshape(p, p)
        V_iiaa_01 = qs.v_term(i, i, a, a, 0, 1).reshape(a, a)
        T_xy = qs.t_term(0, 1)
        V_iipp_10 = qs.v_term(i, i, p, p, 1, 0).reshape(p, p)
        V_iiaa_10 = qs.v_term(i, i, a, a, 1, 0).reshape(a, a)
        T_yx = qs.t_term(1, 0)

        energy += np.sum(V_iiaa_01 * (T_xy)) / 2
        energy += V_iipp_01 @ terms.b_term @ terms.b_term / 2
        energy += np.sum(V_iiaa_10 * (T_yx)) / 2
        energy += V_iipp_10 @ terms.b_term @ terms.b_term / 2

        energy = energy * site
    else:
        # energy calculations
        for x in range(site):
            energy += terms.h_ip @ qs.B_term(x)

            for y in range(site):
                if x < y:
                    V_iipp = qs.v_term(i, i, p, p, x, y).reshape(p, p)
                    V_iiaa = qs.v_term(i, i, a, a, x, y).reshape(a, a)
                    T_xy = qs.t_term(x, y)

                    # noinspection SpellCheckingInspection
                    energy += np.sum(V_iiaa * (T_xy))
                    # noinspection SpellCheckingInspection
                    energy += V_iipp @ qs.B_term(x) @ qs.B_term(y)

    print(iteration)

    return (
        energy,
        tensors.t_a_i_tensor,
        tensors.t_ab_ij_tensor,
        single,
        double,
    )


# if __name__ == "__main__":

#     t_periodic(5, 5, 1, 0, 0, 1e-8, 0.5, 3,  False, 3, True, "sin")
#     print("Hello?")
