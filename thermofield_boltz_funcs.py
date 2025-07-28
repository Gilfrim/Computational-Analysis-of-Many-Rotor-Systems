import numpy as np
import scipy as scp
import opt_einsum as oe

def analytic_boltzmann_factor(state_energies,beta):
    boltzmann_factors = np.exp(-beta*state_energies) # partial state occupation
    
    return(boltzmann_factors)


def analytic_calculations_boltzmann(Energy_eigenvalues,beta):
        # Calculate partition function based on boltzmann dist
        analytic_boltzmann_factors = analytic_boltzmann_factor(Energy_eigenvalues,beta)
        
        # Partition function for a 1-particle system  
        Z_1 = np.sum(analytic_boltzmann_factors) 
        
        # leave room for multiparticle system 
        Z = Z_1
       
        # 1-particle partial occupation of states
        partial_occupations = analytic_boltzmann_factors /Z_1
        
        # calculate thermal internal Energy
        internal_energy = np.sum(Energy_eigenvalues * partial_occupations)

        return partial_occupations, Z, internal_energy


def analytic_results_boltzmann(temp_init,temp_final,Hamiltonian):

    # 1 site state energies 
    state_energies = np.linalg.eigvalsh(Hamiltonian)
    occupation_results = []
    Z_results = []
    internal_energy_results = []
    
    temp_range = np.linspace(temp_init,temp_final,100000)

    for temp in (temp_range):
        beta = 1. / (1 * temp)
        partial_occupations, Z, internal_energy = analytic_calculations_boltzmann(state_energies,beta)
        
        
        occupation_results.append(partial_occupations)
        
        Z_results.append(Z)
        internal_energy_results.append(internal_energy)
    
    return(temp_range,internal_energy_results,occupation_results,Z_results)

def compound_space_z(beta,H_tilde):
    eig_val_x, Vecs_x = np.linalg.eigh(H_tilde)
    V_0 = Vecs_x[:,0]
    Z = np.sum(((V_0)**2)*np.exp(-beta*eig_val_x))
    return(Z)

def thermofield_change_of_basis(hamiltonian):
    # place hamiltonian in tensor product basis 
    d_pq = np.eye(hamiltonian.shape[0],hamiltonian.shape[1])
    primitive_hamiltonian = oe.contract('pq,mw->pmqw', hamiltonian, d_pq,  optimize='optimal').reshape(hamiltonian.shape[0]**2, hamiltonian.shape[0]**2)
    # primitive_hamiltonian = np.kron(hamiltonian,d_pq)

    # Constants  
    physical_hilbert_dim = np.shape(hamiltonian)[0]

    # Create Uniform state vector 1st diagonal basis vector 
    u_vec = (np.ones(physical_hilbert_dim)*physical_hilbert_dim)**-0.5

    # Create set of orthogonal vectors from U to define diagonal basis vectors 
    # This is the diagonal basis vectors which need to be pruned    
    diagonal_vecs = []
    for n in range(physical_hilbert_dim):
        diagonal_vec = -(1/np.sqrt(physical_hilbert_dim))*np.copy(u_vec)
        diagonal_vec[n] +=  1
        diagonal_vecs.append(diagonal_vec)  

    diagonal_vecs = np.array(diagonal_vecs)

    # The diagonal basis is now over-complete we need to use a lowdin procedure 
    overlap_matrix = diagonal_vecs.T@diagonal_vecs
    eig_vals_lowdin,eig_vecs_lowdin = np.linalg.eigh(overlap_matrix)

    # Check for eigenvalues that are zero they are linear dependent and can be removed 
    # this hardcoded because the there should only be 1 excess basis vector and all other eigenvalues will be 1 
    linearly_independent_eigvals = eig_vals_lowdin[1:]
    # print(linearly_independent_eigvals)

    # Remove linear dependent vector from set of eigenvectors  
    # Columns of eigh output are the eigenvectors 
    linearly_independent_vecs = eig_vecs_lowdin[:,1:]

    # Find the reduced basis set 
    # Note: columns are basis vectors  
    # Reduced basis written C_{m\lambda}
    reduced_basis = diagonal_vecs@linearly_independent_vecs

    # Create diagonal state basis including
    # Diagonal basis as C_{m\lambda} 
    diagonal_basis = np.zeros((physical_hilbert_dim,physical_hilbert_dim)) 
    diagonal_basis[:,0] = u_vec
    diagonal_basis[:,1:] = reduced_basis

    # Construct change of basis matrix for primitive tensor prod basis to diagonal tensor prod basis 
    change_of_basis_matrix = np.zeros((primitive_hamiltonian.shape))

    # necessary counter for non diagonal basis vectors in tensor product basis
    counter = 0
    
    primitive_basis_order = []
    for col in range(primitive_hamiltonian.shape[0]):
        for i in range(physical_hilbert_dim):
            for j in range(physical_hilbert_dim):
                
                primitive_basis_order.append((i,j))
                
                # map diagonal vectors to primitive tensor product basis 
                if i == j and col < (primitive_hamiltonian.shape[0])**0.5:
                    change_of_basis_matrix[:,col][i*physical_hilbert_dim+j] = diagonal_basis[i,col] 
                
                # makes sure the rest of the basis vectors from tensor product basis are in change of basis 
                # Ensure there are no diagonal components as diagonal vectors in primitive tensor prod basis is given above 
                elif col >= (primitive_hamiltonian.shape[0])**0.5 and i != j:

                    if not np.any(change_of_basis_matrix[:,col]): # if column after diagonal vectors in tensor basis is zeros    
                        
                        # column is not the first column after diagonal insertion 
                        # and column index is mod divisible by number of elements for diagonal component
                        # we add in additional displacment such that vector added never has diagonal component in primitive tensor basis  
                        if counter > 0 and col % (primitive_hamiltonian.shape[0])**0.5 == 0: 
                            counter += 1 
                        
                        # add tensor prod basis vector in orginal basis set                  
                        change_of_basis_matrix[:,col][i*physical_hilbert_dim+j+counter] = 1
                        
                        # displace vector added to new basis per each column
                        counter += 1 
    
    return(change_of_basis_matrix,primitive_hamiltonian)

def depr_H_tilde_maker(hamiltonian):
    # This H_tilde_maker is deprecated due to the change of basis  
    # The hamiltonian can be written in 4 blocks 
    
    # Constants  
    physical_hilbert_dim = np.shape(hamiltonian)[0]

    ## Define  H_tilde_1 by transforming the hamiltonian to diagonal basis 
    
    # Create Uniform state vector 1st diagonal basis vector 
    u_vec = (np.ones(physical_hilbert_dim)*physical_hilbert_dim)**-0.5

    # Create set of orthogonal vectors from U to define diagonal basis vectors 
    # This is the primitive basis   
    primitive_vecs = []
    for n in range(physical_hilbert_dim):
        primitive_vec = -(1/np.sqrt(physical_hilbert_dim))*np.copy(u_vec)
        primitive_vec[n] +=  1
        primitive_vecs.append(primitive_vec)  

    primitive_vecs = np.array(primitive_vecs)

    # The primitive basis is over complete we need to use a lowdin procedure 
    overlap_matrix = primitive_vecs.T@primitive_vecs
    eig_vals_lowdin,eig_vecs_lowdin = np.linalg.eigh(overlap_matrix)
    
    # Check for eigenvalues that are zero they are linear dependent and can be removed 
    # this hardcoded because the there should only be 1 excess basis vector and all other eigenvalues will be 1 
    linearly_independent_eigvals = eig_vals_lowdin[1:]
    print(linearly_independent_eigvals)

    # Remove linear dependent vector from set of eigenvectors  
    # Columns of eigh output are the eigenvectors 
    linearly_independent_vecs = eig_vecs_lowdin[:,1:]

    # Find the reduced basis set 
    # Note: columns are basis vectors  
    # Reduced basis written C_{m\lambda}
    reduced_basis = primitive_vecs@linearly_independent_vecs
    
    # Create diagonal state basis including
    # Diagonal basis as C_{m\lambda} 
    diagonal_basis = np.zeros((physical_hilbert_dim,physical_hilbert_dim)) 
    diagonal_basis[:,0] = u_vec
    diagonal_basis[:,1:] = reduced_basis
    
    # Place hamiltonian in basis primitive basis 
    # Its diagonal in primitive basis 
    primitive_hamiltonian = np.diag(np.diag(hamiltonian))

    # Write hamiltonian as in primitive tensor product basis
    # diagonal_basis.T as C_{\lambda,m}
    H_Tilde_1 = diagonal_basis.T@primitive_hamiltonian@diagonal_basis
    

    ######################## H tilde 2 and 3  
    H_Tilde_2 = np.zeros((physical_hilbert_dim,physical_hilbert_dim*(physical_hilbert_dim)))

    for i in range(physical_hilbert_dim):
        for j in range(physical_hilbert_dim):
            for k in range(physical_hilbert_dim):
                if j == k:     
                    H_Tilde_2[i,k + physical_hilbert_dim*j] = 0 
                else: 
                    H_Tilde_2[i,k + physical_hilbert_dim*j] = diagonal_basis[j,i]*hamiltonian[j,k] 

    indices_to_keep_H2 = np.ix_([i for i in range(physical_hilbert_dim*physical_hilbert_dim) if (i // physical_hilbert_dim) != (i % physical_hilbert_dim)])[0]
    H_Tilde_2 = H_Tilde_2[:,indices_to_keep_H2]
    H_Tilde_3 = H_Tilde_2.T
    
    
    ######################## Htilde 4     
    d_tilde = np.eye((physical_hilbert_dim))

    h_4 = np.kron(d_tilde,hamiltonian) # The element only exist if the hamiltonian is diagonal in tilde basis  

    # Generate indices to keep (n \neq l_tilde)
    indices_to_keep_H4 = [i for i in range(physical_hilbert_dim*physical_hilbert_dim) if (i // physical_hilbert_dim) != (i % physical_hilbert_dim)]

    # Slice the matrix
    H_tilde_4 = h_4[np.ix_(indices_to_keep_H4, indices_to_keep_H4)]


    # Build H Tilde 
    H_tilde = np.zeros((physical_hilbert_dim**2,physical_hilbert_dim**2))
    H_tilde[:physical_hilbert_dim,:physical_hilbert_dim] = H_Tilde_1
    H_tilde[:physical_hilbert_dim, physical_hilbert_dim:] = H_Tilde_2 
    H_tilde[physical_hilbert_dim:,:physical_hilbert_dim] = H_Tilde_3
    H_tilde[physical_hilbert_dim:,physical_hilbert_dim:] = H_tilde_4
    return(H_tilde)



def H_tilde_maker(hamiltonian):
    change_of_basis_mat, primitive_hamiltonian = thermofield_change_of_basis(hamiltonian)
    H_tilde = change_of_basis_mat.T@primitive_hamiltonian@change_of_basis_mat
    return(H_tilde)

def H_tilde_partition(H_tilde):
    # check that partitions is correct
    H_tilde_dict = { 
        'h_ia': H_tilde[0:1,1:],
        'h_ai': H_tilde[1:,0:1],
        'h_ij': H_tilde[:1,:1],
        'h_ab': H_tilde[1:,1:],
    }
    return(H_tilde_dict)

def T_inits(H_tilde_dict):
    T_0 = 0 
    T_ai = np.zeros((H_tilde_dict['h_ai'].shape))
    return(T_0,T_ai)

def boltzmann_thermofield_prop(dT_ai,H_dict): 
    
    dT_H = np.zeros(np.shape(dT_ai))
    dT_H += H_dict['h_ai']
    dT_H += np.einsum('ac,ci->ai', H_dict['h_ab'], dT_ai)
    dT_H -= np.einsum('ak,ki->ai', dT_ai, H_dict['h_ij'])
    dT_H -= np.einsum('ak,kc,ci->ai', dT_ai,H_dict['h_ia'],dT_ai)
      
    dT_ai = dT_H  
    return (dT_ai)

def euler_solver(T_initial,T_final,number_of_steps,H):
 
    # initialize the temperature values for graphing
    kb = 1
    beta_final = 1/(kb*T_final)
    dtau = beta_final/number_of_steps 
    beta_values = np.linspace(dtau, beta_final, number_of_steps) 
    
    # initial partition func 
    t_0 = T_initial[0] 
    T_ai = T_initial[1]

    real_temp_values = 1. / (kb * beta_values)
    
    partition_results = []
    internal_energy_results = []
    
    for i in range(number_of_steps):
        # integration kernel
        dT_ai = boltzmann_thermofield_prop(T_ai,H)
        t_0 -= dtau*((np.trace(H['h_ij']) +  np.einsum('ia,ai->',H['h_ia'],T_ai)))
        T_ai -= dtau*dT_ai 
    
        # properties 
        U_t = (np.trace(H['h_ij']) +  (np.einsum('ia,ai->',H['h_ia'],T_ai)))
        
        # Append results
        partition_results.append(np.exp(t_0))
        internal_energy_results.append(U_t)
        
    return(real_temp_values,internal_energy_results,partition_results)