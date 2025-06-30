import numpy as np
import matplotlib.pyplot as plt
import quant_rotor.models.thermofield_boltz_funcs as bz


# hamiltonian = np.array([[7,9,2,5],[0,2,3,5],[0,0,3,2],[0,0,2,2]]) # good 
# hamiltonian = np.array([[7,9,7],[1,0,2],[8,2,3]]) # good
# hamiltonian = np.array([[7,9,1,9],[0,2,1,6],[9,0,9,2],[1,3,6,2]]) # bad
hamiltonian = np.array([[7,9,1,9,0],[0,2,1,6,8],[9,0,9,2,1],[1,3,6,2,7],[12, 2, 4,1,15]]) # bad


hamiltonian = (hamiltonian+hamiltonian.T)/2
vals, vecs = np.linalg.eigh(hamiltonian)


diag_new = np.diag(vals - np.min(vals))
hamiltonian = vecs@diag_new@vecs.T



physical_hilbert_space_dim = hamiltonian.shape[0]

H_tilde = bz.H_tilde_maker(hamiltonian)
temps, int_eng_anal, occ_anal, Z_anal = bz.analytic_results_boltzmann(0.5,200,hamiltonian,H_tilde)


H_tilde = bz.H_tilde_partition(H_tilde)


T_0_init, T_ai_init = bz.T_inits(H_tilde) 
T_init = (T_0_init,T_ai_init)

real_temp_values,internal_energy_results,partition_results = bz.euler_solver(T_init,0.5,10000,H_tilde)

# print(real_temp_values[-1])
# print(temps[0])
import matplotlib.pyplot as plt
import numpy as np

# Create a figure with 2 subplots (1 row, 2 columns)
fig, axs = plt.subplots(2, 1, figsize=(10, 10),sharex=True)


# --- First subplot: Internal Energy ---
axs[0].plot(temps, int_eng_anal, '1', label='Analytical', markevery=1500)
axs[0].plot(real_temp_values, internal_energy_results, label='Thermofield')
axs[0].set_title('Internal Energy for Single Site Rotor')
axs[0].set_ylabel('U/kb (K)', fontsize=14)
axs[0].set_xlabel('Temperature (K)')
axs[0].set_xlim(0, 200)
axs[0].tick_params(axis='both', which='major', labelsize=12)
axs[0].legend()

# --- Second subplot: Partition Function ---
axs[1].plot(temps, Z_anal, '1', label='Analytical', markevery=1500)
axs[1].plot(real_temp_values, np.array(partition_results) * physical_hilbert_space_dim, label='Thermofield')
axs[1].set_title('Partition function for Single Site Rotor')
axs[1].set_ylabel('Z', fontsize=14)
axs[1].set_xlabel('Temperature (K)')
axs[1].set_xlim(0, 200)
axs[1].tick_params(axis='both', which='major', labelsize=12)
axs[1].legend()

plt.tight_layout()
plt.show()



# plt.plot(real_temp_values,internal_energy_results)
# plt.show()
# plt.plot(real_temp_values,partition_results)
# plt.show()