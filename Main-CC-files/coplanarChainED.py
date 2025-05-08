import numpy as np
import hamiltonianGenerator as hg


np.set_printoptions(suppress = True, linewidth = 200, threshold = 121*121, precision = 6)

# Script for diagonalizing the Hamiltonian for 2 dipolar rotors oriented along the x direction. 
# m_max = 5 is sufficient for convergence when studying ground state properties. 
m_max = 2
d = 2*m_max + 1
K_1 = np.zeros((d,d))
for i in range(d):
    K_1[i,i] = hg.free_one_body(i, i, m_max)

K_2 = np.kron(K_1, np.eye(d)) + np.kron(np.eye(d), K_1)

# Interaction for a coplanar chain is proportional to (yiyj + 2xixj)/r^3. If sites are not coplanar, 
# the interaction will be some other linear combination of xixj, yiyj, (xiyj + xjyi). The coefficients 
# will, in general, depend on the angle as well as the length of the vector connecting the sites. 
V2 = np.zeros((d**2, d**2))

for i in range(d):
    for j in range(d):
        for k in range(d):
            for l in range(d):
                V2[i*d + j,k*d + l] = hg.interaction_yiyj(i, j, k, l) + 2*hg.interaction_xixj(i, j, k, l)
'''
num_pts = 20
E0_array = np.zeros((num_pts, 2))
for i in range(num_pts):
    g = (i+1.0)/num_pts
    H_2 = K_2 + g * V_2
    evals, evecs = np.linalg.eigh(H_2)
    E0 = evals[0]
    E0_array[i, 0] = g
    E0_array[i, 1] = E0

np.savetxt("E0_N2_ED.csv", E0_array, delimiter=",", header="g, E0")'''


print(V2)