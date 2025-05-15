import numpy as np
from itertools import product
from collections import defaultdict
import time
import functions as func
import hamiltonianGenerator as hG
import hamiltonian_Bryan as ham

#alternate version for hamiltonian that doesn't use csv files and directly calls shaeer's rotor functions

# #printout formatting for large matrices
# np.set_printoptions(suppress = True, linewidth = 1500, threshold = 10000, precision = 6)

def hamiltonian(site_total:int, state_total:int, g:float = 1, onesite:bool = True, twosite:bool = True, timer:bool = True)->np.array:
    """Docstring"""
    start = time.perf_counter()
    m_max = abs(func.p_to_m(state_total - 1))
    dim = state_total ** site_total
    H = np.zeros((dim, dim))

    p_to_m_dict = {p: func.p_to_m(p) for p in range(state_total)}

    if onesite:
        location_dict_1 = list(defaultdict(list) for _ in range(site_total))

        for index_1, state_1 in enumerate(func.state_generator(state_total, site_total)):
            for onesite_index in range(site_total):
                key_1 = tuple(state_1[i] for i in range(site_total) if i != onesite_index)
                location_dict_1[onesite_index][key_1].append((index_1, state_1[onesite_index]))

        for onesite_hamiltonian_term in range(site_total):
            for keys_1, points_1 in location_dict_1[onesite_hamiltonian_term].items():
                for row_1, col_1 in product(points_1, repeat = 2):
                    H[row_1[0], col_1[0]] += hG.free_one_body(p_to_m_dict[row_1[1]] + m_max, p_to_m_dict[col_1[1]] + m_max, m_max)

    if twosite:
        location_dict_2 = list(defaultdict(list) for _ in range(site_total))

        for index_2, state_2 in enumerate(func.state_generator(state_total, site_total)):
            for twosite_index in range(site_total):
                key_2 = tuple(state_2[j] for j in range(site_total) if j != (twosite_index % site_total) and j != ((twosite_index + 1) % site_total))
                location_dict_2[twosite_index][key_2].append((index_2, state_2[twosite_index], state_2[(twosite_index + 1) % site_total]))

        for twosite_hamiltonian_term in range(site_total):
            for keys_2, points_2 in location_dict_2[twosite_hamiltonian_term].items():
                for row_2, col_2 in product(points_2, repeat = 2):
                    H[row_2[0], col_2[0]] += g * hG.interaction_two_body_coplanar(p_to_m_dict[row_2[1]] + m_max, p_to_m_dict[row_2[2]] + m_max, p_to_m_dict[col_2[1]] + m_max, p_to_m_dict[col_2[2]] + m_max)

    end = time.perf_counter()
    if timer:
        print("The time of execution of above program is :", (end - start), "s")

    return H


if __name__ == "__main__":
    site = 3
    state = 3
    g = 1
    hami = hamiltonian(site, state, timer = False)

    print(hami[1])

    #vals, vecs = np.linalg.eigh(hami)
    #print(vals)

    # what = ham.hamiltonian(site, state, timer = False)
    # print(np.all(hami == what))

    # thing = state ** (site - 1)
    # diag = np.diag(ham)
    # print(diag)
    # for what in range(state):
    #     print(diag[what * thing])

