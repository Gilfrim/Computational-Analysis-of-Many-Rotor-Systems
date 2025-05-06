import numpy as np
from itertools import product
from collections import defaultdict
import time

#printout formatting for large matrices
np.set_printoptions(suppress = True, linewidth = 200, threshold = 1000)

#how many sites and how many states per site
#site_total = int(input("Input number of sites: "))
#state_total = int(input("Input number of states per site: "))
site_total = 4
state_total = 4
little_h = []
for i in range(site_total):
    little_h.append((i + 1) * np.eye(site_total))

q = len(little_h)

#Start timer to find how long it takes to put values in hamiltonian
start = time.time()

#creates list of tuples containing all state combinations
states = list(product(range(state_total), repeat = site_total))

#dimension of hamiltonian
dim = len(states)

#creates matrix with size dim * dim(little_h )x dim * dim(little_h) (for tensor product) with zero entries to be filled in later
#dimension of H will be [dim(little_h)][state_total**site_total]
H = np.zeros((dim * q, dim * q))
'''
H_0 = np.zeros((dim, dim))
H_1 = np.zeros((dim, dim))
H_2 = np.zeros((dim, dim))
listboi = [H_0, H_1, H_2]
'''


#creates a dictionary for each onesite hamiltonian h_P, h_Q,....
grouped_delta = list(defaultdict(list) for i in range(site_total))

#creates the key (which terms will match for kronecker delta functions of that row)
#adds the column number where the delta functions match to a list matched with that rows key
for index, state in enumerate(states):
    for onesite_hamiltonian_index in range(site_total):
        key = tuple(state[i] for i in range(site_total) if i != onesite_hamiltonian_index)
        grouped_delta[onesite_hamiltonian_index][key].append(index)

'''
for onesite_hamiltonian_term in range(site_total):
    for key, indices in grouped_delta[onesite_hamiltonian_term].items():
        for row in indices:
            for col in indices:
                listboi[onesite_hamiltonian_term][row, col] += 1
'''

tens_start = time.time()

#(site_total)^4(state_total^(site_total))^2
for whatever in range(site_total):
    for key_1, indices_1 in grouped_delta[whatever].items():
        for row_1 in indices_1:
            for col_1 in indices_1:
                for i in range(q):
                    for j in range(q):
                        H[row_1 * q + i, col_1 * q + j] += little_h[whatever][i, j]

tens_end = time.time()
'''
#print(grouped_delta)
a = np.kron(listboi[0], little_h[0])
b = np.kron(listboi[1], little_h[1])
c = np.kron(listboi[2], little_h[2])

d = np.add(a, b)
e = np.add(d, c)

#print(e)
#print(np.all(H == e))
'''

end = time.time()
print(H)

print("The time of execution of above program is :", (end - start), "s")
print("The time of execution for the tensor product is :", (tens_end - tens_start), "s")
print("The percentage of time for tensor product is :", (tens_end - tens_start) / (end-start) * 100, "%")

#np.linalg.eig(H) time complexity a * (N**3) = time (possible a value = 6.23986*(10**(-10))
#dimension max currently site = 5, state = 5 (exclusive)
diag_start = time.time()
vecs, vals = np.linalg.eigh(H)
sorted_arr = np.sort(vecs, axis=None)[::-1]
diag_end = time.time()
print(sorted_arr)
print("The time of execution of computing eigenvalues and eigenvectors is :", (diag_end - diag_start), "s")