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

    


if __name__ == "__main__":
    site = 3
    state = 3
    gee = 10
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

