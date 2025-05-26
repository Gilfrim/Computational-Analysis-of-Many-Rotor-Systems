import numpy as np
import time
from itertools import product

np.set_printoptions(suppress = True, linewidth = 500, threshold = 100000, precision = 10)

start = time.time()

site_total = 3
state_total = 2

dim = state_total ** site_total
identity = np.identity(state_total)
H = np.zeros((dim, dim))

basis = []
for basis_vector in range(state_total):
    basis.append(np.zeros(state_total))
    basis[basis_vector][basis_vector] = 1


#e^{P}_{P'}(x) =ket{P}bra{P'}(x)
#for state in range(site_total):
for state in range(site_total):
    for i in range(state_total):
        for j in range(state_total):
            proj = np.outer(basis[i],basis[j])
            tensor_prod_list = []
            for position in range(site_total):
                if position == state:
                    tensor_prod_list.append(proj)
                else:
                    tensor_prod_list.append(identity)
            tensor_prod = tensor_prod_list[0]
            for repeat in range(site_total - 1):
                tensor_prod = np.kron(tensor_prod, tensor_prod_list[repeat + 1])
            H += (10 ** state) * tensor_prod


end = time.time()


# for a in range(state_total):
#     for b in range(state_total):
#         for c in range(state_total):
#             for d in range(state_total):
#                 proj_p_2_ab = np.outer(basis[a], basis[b])
#                 pad_p_2_ab = np.kron(proj_p_2_ab, np.kron(identity, identity))
#
#                 proj_q_2_cd = np.outer(basis[c], basis[d])
#                 pad_q_2_cd = np.kron(identity, (np.kron(proj_q_2_cd, identity)))
#
#                 e = np.matmul(pad_p_2_ab, pad_q_2_cd)
#                 H += 3 * e
#
#                 proj_q_2_ab = np.outer(basis[a], basis[b])
#                 pad_q_2_ab = np.kron(identity, (np.kron(proj_q_2_ab, identity)))
#
#                 proj_r_2_cd = np.outer(basis[c], basis[d])
#                 pad_r_2_cd = np.kron(identity, (np.kron(identity, proj_r_2_cd)))
#
#                 e = np.matmul(pad_q_2_ab, pad_r_2_cd)
#                 H += 30 * e
#
#                 proj_r_2_ab = np.outer(basis[a], basis[b])
#                 pad_r_2_ab = np.kron(identity, (np.kron(identity, proj_r_2_ab)))
#
#                 proj_p_2_cd = np.outer(basis[c], basis[d])
#                 pad_p_2_cd = np.kron(proj_p_2_cd, np.kron(identity, identity))
#
#                 e = np.matmul(pad_r_2_ab, pad_p_2_cd)
#                 H += 300 * e


states = list(product(range(state_total), repeat = site_total))

# H_t = np.zeros((dim, dim))
#
# location_dict_1 = list(defaultdict(list) for i in range(site_total))
#
# for index_1, state_1 in enumerate(states):
#     for onesite_index in range(site_total):
#         key_1 = tuple(state_1[i] for i in range(site_total) if i != onesite_index)
#         location_dict_1[onesite_index][key_1].append(index_1)
#
# for onesite_hamiltonian_term in range(site_total):
#     for keys_1, points_1 in location_dict_1[onesite_hamiltonian_term].items():
#         for row_1 in points_1:
#             for col_1 in points_1:
#                 H_t[row_1, col_1] += 10 ** onesite_hamiltonian_term


print(H)
#print(np.all(H == H_t))
#print("The time of execution of above program is :", (end-start), "s")


