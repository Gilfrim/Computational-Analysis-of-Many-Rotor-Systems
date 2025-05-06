import numpy as np
from itertools import product
from collections import defaultdict

import csv

#printout formatting for large matrices
np.set_printoptions(suppress = True, linewidth = 1500, threshold = 10000, precision = 9)

def p_to_m(num: int) -> int:
    """Converts p (0, 1, 2, 3,...) to m (0, -1, 1, -2,...)"""
    return num // 2 if num % 2 == 0 else -((num + 1) // 2)

def hamiltonian(sites:int, states:int ,g:float = 1, onesite:bool = True, twosite:bool = True)->np.array:
    """Docstring"""

    m_max = abs(p_to_m(states - 1))
    m_max_shaeer = 5

    states_base_site = list(product(range(states), repeat = sites))
    dim = states ** sites
    H = np.zeros((dim, dim))

    if onesite:
        location_dict_1 = list(defaultdict(list) for _ in range(sites))

        for index_1, state_1 in enumerate(states_base_site):
            for onesite_index in range(sites):
                key_1 = tuple(state_1[i] for i in range(sites) if i != onesite_index)
                location_dict_1[onesite_index][key_1].append((index_1, state_1[onesite_index]))

        h_dict_1 = defaultdict(float)
        with open(r"h_values", mode = "r", newline = "") as csvfile_h_1:
            reader_h_1 = csv.reader(csvfile_h_1, delimiter = ",")
            next(reader_h_1)
            for row_h in reader_h_1:
                h_dict_1[int(row_h[0]), int(row_h[1])] = float(row_h[2])

        for onesite_hamiltonian_term in range(sites):
            for keys_1, points_1 in location_dict_1[onesite_hamiltonian_term].items():
                for row_1 in points_1:
                    for col_1 in points_1:
                        H[row_1[0], col_1[0]] += h_dict_1.get((row_1[1], col_1[1]), 0)


    if twosite:
        location_dict_2 = list(defaultdict(list) for _ in range(sites))
        for index_2, state_2 in enumerate(states_base_site):
            # for twosite_index in range(sites):
            #     key_2 = tuple(state_2[j] for j in range(sites) if j != twosite_index and (sites > twosite_index + 1 != j))
            #     if twosite_index + 1 != sites:
            #         location_dict_2[twosite_index][key_2].append((index_2, state_2[twosite_index], state_2[(twosite_index + 1)]))
            for twosite_index in range(sites):
                key_2 = tuple(state_2[j] for j in range(sites) if j != (twosite_index % sites) and j != ((twosite_index + 1) % sites))
                location_dict_2[twosite_index][key_2].append((index_2, state_2[twosite_index], state_2[(twosite_index + 1) % sites]))

        h_dict_2 = defaultdict(float)
        with open(r"v_values", mode = "r", newline = "") as csvfile_v_1:
            reader_v_1 = csv.reader(csvfile_v_1, delimiter = ",")
            next(reader_v_1)
            for row_v in reader_v_1:
                h_dict_2[int(row_v[0]), int(row_v[1]), int(row_v[2]), int(row_v[3])] = float(row_v[4])

        for twosite_hamiltonian_terms in range(sites):
            for keys_2, points_2 in location_dict_2[twosite_hamiltonian_terms].items():
                for row_2 in points_2:
                    for col_2 in points_2:
                        H[row_2[0], col_2[0]] += (h_dict_2.get((row_2[1], row_2[2], col_2[1], col_2[2]), 0))

    return H

# if __name__ == "__main__":
#     site = 2
#     state = 3
#     gee = 0.2
#     ham = hamiltonian(site, state, gee)
#     print(ham)
#     vals, vecs = np.linalg.eigh(ham)
#     #print(vals)






