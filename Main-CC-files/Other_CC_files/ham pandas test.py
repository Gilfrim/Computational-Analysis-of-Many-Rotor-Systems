import numpy as np
from itertools import product
from collections import defaultdict
import time
import pandas as pd
import functions as func

#printout formatting for large matrices
np.set_printoptions(suppress = True, linewidth = 1500, threshold = 10000, precision = 6)

def hamiltonian(site_total: int, state_total: int, onesite:bool = True, twosite:bool = True, timer:bool = True)->np.array:
    """Docstring"""
    start = time.perf_counter()
    m_max_shaeer = 5
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

        h_dict_1 = defaultdict(float)
        file_k = pd.read_csv(r"C:\Users\Bryan\Desktop\Coop\Shaeer code\MatrixElementGenerator\matrix_elements_K.csv", skiprows = [0, 1], delimiter = ",", names = ["m_1", "m_2", "value"])
        for _, row_k in file_k.iterrows():
            m_1_k = int(row_k["m_1"])
            m_2_k = int(row_k["m_2"])
            value_k = float(row_k["value"])
            if value_k != 0:
                h_dict_1[((m_1_k - m_max_shaeer), (m_2_k - m_max_shaeer))] = value_k

        for onesite_hamiltonian_term in range(site_total):
            for keys_1, points_1 in location_dict_1[onesite_hamiltonian_term].items():
                for row_1, col_1 in product(points_1, repeat = 2):
                    H[row_1[0], col_1[0]] += h_dict_1.get((p_to_m_dict[row_1[1]], p_to_m_dict[col_1[1]]), 0)

    if twosite:
        location_dict_2 = list(defaultdict(list) for _ in range(site_total))

        for index_2, state_2 in enumerate(func.state_generator(state_total, site_total)):
            for twosite_index in range(site_total):
                key_2 = tuple(state_2[j] for j in range(site_total) if j != (twosite_index % site_total) and j != ((twosite_index + 1) % site_total))
                location_dict_2[twosite_index][key_2].append((index_2, state_2[twosite_index], state_2[(twosite_index + 1) % site_total]))

        h_dict_2 = defaultdict(float)
        file_v = pd.read_csv(r"C:\Users\Bryan\Desktop\Coop\Shaeer code\MatrixElementGenerator\matrix_elements_V.csv", skiprows = [0, 1], delimiter = ",", names = ["m_1", "m_2", "m_3", "m_4", "value"])
        for _, row_v in file_v.iterrows():
            m_1_v = int(row_v["m_1"])
            m_2_v = int(row_v["m_2"])
            m_3_v = int(row_v["m_3"])
            m_4_v = int(row_v["m_4"])
            value_v = float(row_v["value"])
            if value_v != 0:
                h_dict_2[((m_1_v - m_max_shaeer), (m_2_v - m_max_shaeer), (m_3_v - m_max_shaeer), (m_4_v - m_max_shaeer))] = value_v
                if int(m_1_v != m_3_v) and int(m_2_v != m_4_v):
                    h_dict_2[((m_3_v - m_max_shaeer), (m_4_v - m_max_shaeer), (m_1_v - m_max_shaeer), (m_2_v - m_max_shaeer))] = value_v

        for twosite_hamiltonian_terms in range(site_total):
            for keys_2, points_2 in location_dict_2[twosite_hamiltonian_terms].items():
                for row_2, col_2 in product(points_2, repeat = 2):
                    H[row_2[0], col_2[0]] += 2*(h_dict_2.get((p_to_m_dict[row_2[1]], p_to_m_dict[row_2[2]], p_to_m_dict[col_2[1]], p_to_m_dict[col_2[2]]), 0))

    end = time.perf_counter()
    if timer:
        print("The time of execution of above program is :", (end - start), "s")

    return H

if __name__ == "__main__":
    site = 5
    state = 9
    ham = hamiltonian(site, state)
    #print(ham)
    #ha = hamilton.hamiltonian(site, state, timer = False)
    #print(ha)
    #print(np.all(ham == ha))
    # val, vec = np.linalg.eigh(ham)
    # print(val)