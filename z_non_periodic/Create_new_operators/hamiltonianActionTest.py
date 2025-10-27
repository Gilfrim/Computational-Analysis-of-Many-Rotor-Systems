import numpy as np
import hamiltonianGenerator as hg

m_max = 1
d = 2*m_max + 1
K = np.zeros((d,d))
for i in range(d):
    v_i = np.zeros(d)
    v_i[i] = 1.0
    for j in range(d):
        v_j = np.zeros(d)
        v_j[j] = 1.0
        w = np.zeros(d)
        K[i,j] = np.matmul(np.transpose(v_i),hg.free_action_one_body(m_max, v_j, w))

l = d**2
V = np.zeros((l,l))
for i in range(l):
    for j in range(l):
        v_i = np.zeros(l)
        v_i[i] = 1.0
        v_j = np.zeros(l)
        v_j[j] = 1.0
        w = np.zeros(l)
        V[i,j] = np.matmul(np.transpose(v_i),hg.interaction_action_two_body_coplanar(d, v_j, w))

H = np.kron(np.eye(d),K) + np.kron(K,np.eye(d)) + V
print(H)

K_1 = np.zeros((d,d))
for i in range(d):
    K_1[i,i] = hg.free_one_body(i, i, m_max)

K_2 = np.kron(K_1, np.eye(d)) + np.kron(np.eye(d), K_1)

V_2 = np.zeros((d**2, d**2))
for i in range(d):
    for j in range(d):
        for k in range(d):
            for l_p in range(d):
                V_2[i*d + j,k*d + l_p] += hg.interaction_xixj(i, j, k, l_p) - 2*hg.interaction_yiyj(i, j, k, l_p)

H_2 = K_2 + V_2
print(H_2)

for i in range(l):
    for j in range(l):
        if np.abs(H[i,j] - H_2[i,j]) > 1e-20:
            print("Elements are not equal")

print("Elements are equal")
        

