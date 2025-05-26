import numpy as np
import hamiltonian as ham
import Non_Periodic as non_peri
import ham_new_basis as ham_test
import Periodic as peri

np.set_printoptions(suppress = True, linewidth = 15000, threshold = 10000, precision = 21)

#does a basis transformation on the exact diagonalization hamiltonian

sites = 3
states = 3
g = 0.05

just_onesite = False
just_twosite = False

h_old_h = ham.hamiltonian(sites, states, g, timer = False, twosite = False)
h_old_v = ham.hamiltonian(sites, states, g, timer = False, onesite = False)
h_old = h_old_h + h_old_v

h_original = non_peri.h_term(states, states, 0)
print(h_original)
# h_off = h_original.copy()
#
# for p_off in range(states):
#     for q_off in range(states):
#         if p_off != q_off:
#             h_off[p_off, q_off] = 1 / abs(p_off - q_off)
#
# _, U = np.linalg.eigh(h_off)

matrix = np.zeros((states, states))

for thing in range(states - 1):
    matrix[thing, thing + 1] = 0.5
    matrix[thing + 1, thing] = 0.5

vals, U = np.linalg.eigh(matrix)

unitary = np.allclose(U @ U.T, np.identity(states), atol = 1e-15)

print(f"Unitary test:\n{unitary}")

if not just_onesite and not just_twosite:
    h_trans = U.T @ h_original @ U
elif just_onesite:
    h_trans = U.T @ h_original @ U
elif just_twosite:
    h_trans = np.zeros((states, states))

v_original = non_peri.v_term(states, states, states, states, 0, 1)

if not just_onesite and not just_twosite:
    v_trans = np.einsum("ip, jr, prqs, qk, sl->ijkl", U.T, U.T, v_original, U, U)
elif just_twosite:
    v_trans = np.einsum("ip, jr, prqs, qk, sl->ijkl", U.T, U.T, v_original, U, U)
elif just_onesite:
    v_trans = np.zeros((states, states, states, states))

file_path_h = r"h_values"
with open(file_path_h, mode = "w") as file_h:
    file_h.write("p, q, value\n")
    for row_h in range(states):
        for col_h in range(states):
            file_h.write(f"{row_h}, {col_h}, {h_trans[row_h, col_h]}\n")

file_path_v = r"v_values"
with open(file_path_v, mode = "w") as file_v:
    file_v.write("p, q, r, s, value\n")
    for axis_1 in range(states):
        for axis_2 in range(states):
            for axis_3 in range(states):
                for axis_4 in range(states):
                    file_v.write(f"{axis_1}, {axis_2}, {axis_3}, {axis_4}, {v_trans[axis_1, axis_2, axis_3, axis_4]}\n")
                    #file_v.write(f"{axis_1}, {axis_2}, {axis_3}, {axis_4}, {v_original[axis_1, axis_2, axis_3, axis_4]}\n")

h_new = ham_test.hamiltonian(sites, states)
if not just_onesite and not just_twosite:
    vals_old, _ = np.linalg.eigh(h_old)
elif just_onesite:
    vals_old, _ = np.linalg.eigh(h_old_h)
elif just_twosite:
    vals_old, _ = np.linalg.eigh(h_old_v)

vals_new, _ = np.linalg.eig(h_new)

sorted_old = np.sort(vals_old)
sorted_new = np.sort(vals_new)

print("Eigenvalues:")
print(f"old: {sorted_old}")
print(f"new: {sorted_new}")
print("Eigenvalue difference:")
difference = [abs(old - new) for old, new in zip(sorted_old, sorted_new)]
for index, diff in enumerate(difference):
    print(f"Index {index}: {diff}")

# percent_diff = [((old / new) * 100) - 100 for old, new in zip(sorted_old, sorted_new)]
# for index, diff in enumerate(percent_diff):
#     print(f"Index {index}: {diff}")