import numpy as np
from scipy.sparse.linalg import eigsh
from itertools import product
from collections import defaultdict
import time
import csv
import functions as func

#printout formatting for large matrices
np.set_printoptions(suppress = True, linewidth = 1500, threshold = 10000, precision = 9)

def hamiltonian(sites:int, states:int, mix_factor:float = 0, g:float = 1,  onesite:bool = True, twosite:bool = True, timer = False)->np.array:
    """Calculates full quantum rotor hamiltonian in pqrs basis from Shaeer's m basis rotor generator code"""

    file_path = r"/Users/gilfrim/Desktop/QuantumChemistryCoop/Main-CC-files"

    # Start timer to find how long it takes to put values in hamiltonian
    start = time.perf_counter()

    #m_max could be used in place of m_max_shaeer if Shaeer's code is changed to accommodate
    m_max = abs(func.p_to_m(states - 1))
    m_max_shaeer = 5

    #creates list of tuples containing all state combinations
    #example 3 site 3 state (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0),..., (2, 2, 2)
    #(P, Q, R)
    states_base_site = list(product(range(states), repeat = sites))

    #dimension of hamiltonian
    dim = states ** sites

    #creates matrix with size dim x dim with zero entries to be filled in later
    H = np.zeros((dim, dim))

    if onesite:
        #location_dict_1 is dictionaries for onesite terms, location_dict_2 for twosite
        #creates list of dictionary where each dictionary contains points for a specific hamiltonian term examples h^P_{P'}, h^Q_{Q'},.....
        #or h_{PQ}_{P'Q'}, h^{QR}_{Q'R'},..... for twosite
        location_dict_1 = list(defaultdict(list) for _ in range(sites))

        #mix factor will be 0 by default so this won't be used unless giving a value
        # ⟨m|cosφ|n⟩ = 0.5 * δ_{m, n+1} + 0.5 * δ_{m, n-1}
        if mix_factor != 0:
            matrix = np.zeros((states, states))

            for off_diag in range(states - 1):
                matrix[off_diag, off_diag + 1] = 0.5
                matrix[off_diag + 1, off_diag] = 0.5

        #the dictionary key will be the condition the kronecker deltas must satisfy, and the points for that key are the column
        #example key (0, 0) for h^{P}_{P'},  Q and R values must both be 0
        #containing ((0, P), (9, P), (18, P)) would account for points (0, 0), (0, 9), (0, 18), (9, 0),..., (18, 18)
        #the P is the current state for that point used to convert to m later
        #for onesite
        for index_1, state_1 in enumerate(states_base_site):
            for onesite_index in range(sites):
                key_1 = tuple(state_1[i] for i in range(sites) if i != onesite_index)
                location_dict_1[onesite_index][key_1].append((index_1, state_1[onesite_index]))

        #dictionary for all non-zero values from shaeer K matrix
        #key is (m_1, m_2)
        #shaeer code offsets m value by shaeer_m_max (normally 5)
        h_dict_1 = defaultdict(float)
        with open(file_path + r"/matrix_elements_K.csv", mode = "r", newline = "") as csvfile_h_1:
            reader_h_1 = csv.reader(csvfile_h_1, delimiter = ",")
            #skips first two lines which are formatting and don't contain values
            next(reader_h_1)
            next(reader_h_1)
            for row_h in reader_h_1:
                if float(row_h[2]) != 0.0:
                    #shaeer's code shifts m values to 0 to 2 * m_max, this shifts it back to -m_max to m_max
                    h_dict_1[((int(row_h[0]) - m_max_shaeer), (int(row_h[1]) - m_max_shaeer))] = float(row_h[2])
                    #his code only does upper diagonal elements however it is currently only diagonal elements so this could be simplified
        print(h_dict_1)
        #goes through onesitees to plot poin dictionarits in matrix
        for onesite_hamiltonian_term in range(sites):
            for keys_1, points_1 in location_dict_1[onesite_hamiltonian_term].items():
                for row_1 in points_1:
                    for col_1 in points_1:
                        #adds ⟨m|cosφ|n⟩ to onesite terms
                        if mix_factor != 0:
                            base_val = h_dict_1.get((func.p_to_m(row_1[1]), func.p_to_m(col_1[1])), 0)
                            extra_val = 0
                            for cos_row in range(states):
                                for cos_col in range(states):
                                    extra_val += mix_factor * matrix[cos_row, cos_col] * (1 if (func.p_to_m(cos_col), func.p_to_m(cos_col)) == (func.p_to_m(row_1[1]), func.p_to_m(col_1[1])) else 0)
                            H[row_1[0], col_1[0]] += base_val + extra_val
                        else:
                            H[row_1[0], col_1[0]] += h_dict_1.get((func.p_to_m(row_1[1]), func.p_to_m(col_1[1])), 0)

    print("Here is H:")
    print(H)

    H_New = np.zeros(H.shape)

    if twosite:
        location_dict_2 = list(defaultdict(list) for _ in range(sites))

        # for twosite
        for index_2, state_2 in enumerate(states_base_site):

            for twosite_index in range(sites):
                key_2 = tuple(state_2[j] for j in range(sites) if j != (twosite_index % sites) and j != ((twosite_index + 1) % sites))
                location_dict_2[twosite_index][key_2].append((index_2, state_2[twosite_index], state_2[(twosite_index + 1) % sites]))

                # key_2 = tuple(state_2[j] for j in range(sites) if j != twosite_index and (sites > twosite_index + 1 != j))
                # if twosite_index + 1 != sites:Ok 
                #     location_dict_2[twosite_index][key_2].append((index_2, state_2[twosite_index], state_2[(twosite_index + 1)]))

        # dictionary for all non-zero values from shaeer V tensor
        # key is (m_1, m_2, m_3, m_4)
        # shaeer code offsets m value by shaeer_m_max (normally 5)
        h_dict_2 = defaultdict(float)
        with open(file_path + r"/matrix_elements_V.csv", mode = "r", newline = "") as csvfile_v_1:
            reader_v_1 = csv.reader(csvfile_v_1, delimiter = ",")
            next(reader_v_1)
            next(reader_v_1)
            for row_v in reader_v_1:
                if float(row_v[4]) != 0.0:
                    h_dict_2[((int(row_v[0]) - m_max_shaeer), (int(row_v[1]) - m_max_shaeer), (int(row_v[2]) - m_max_shaeer), (int(row_v[3]) - m_max_shaeer))] = float(row_v[4])
                    # only upper diagonal elements are included in csv file this accounts adds lower diagonal elements
                    # switches 0 -> 2, 1 -> 3
                    if int(row_v[0] != int(row_v[2])) and int(row_v[1] != int(row_v[3])):
                        h_dict_2[((int(row_v[2]) - m_max_shaeer), (int(row_v[3]) - m_max_shaeer), (int(row_v[0]) - m_max_shaeer), (int(row_v[1]) - m_max_shaeer))] = float(row_v[4])

        #goes through twosite dictionaries to plot points in matrix
        for twosite_hamiltonian_terms in range(sites):
            for keys_2, points_2 in location_dict_2[twosite_hamiltonian_terms].items():
                for row_2 in points_2:
                    for col_2 in points_2:
                        H_New[row_2[0], col_2[0]] += g*(h_dict_2.get((func.p_to_m(row_2[1]), func.p_to_m(row_2[2]), func.p_to_m(col_2[1]), func.p_to_m(col_2[2])), 0))
    print("This is H_New:")
    print(H_New)


    end = time.perf_counter()
    if timer:
        print(f"The time of execution of above program is: {end - start}s")

    return H + H_New

if __name__ == "__main__":
    site = 3
    state = 2
    gee = 1
    mix = -0.05
    ham = hamiltonian(site, state, mix, gee)
    print(ham)
    # diag_start = time.perf_counter()
    vals, vecs = np.linalg.eigh(ham)
    # diag_end = time.perf_counter()
    # print(f"The time to diagonalize is: {diag_end - diag_start}s")
    print(np.sort(vals))
    #fast_start = time.perf_counter()
    #lowest_eigenvalue = eigsh(ham, k = 10, which = 'SA', return_eigenvectors = False)
    #fast_end = time.perf_counter()
    #print(f"The time to diagonalize (faster) is: {fast_end - fast_start}s")
    #print(lowest_eigenvalue)

    # print(f"The difference between diagonalization methods is: {np.sort(vals)[0] - lowest_eigenvalue[0]}")




