import numpy as np

def K_tilde_maker(hamiltonian):
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
    # linearly_independent_eigvals = eig_vals_lowdin[1:]
    # print(linearly_independent_eigvals)

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

    print(diagonal_basis.shape)

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

def V_tilde_maker(hamiltonian):
    # The hamiltonian can be written in 4 blocks 
    
    # Constants  
    physical_hilbert_dim = np.shape(hamiltonian)[0]
    d = int(np.sqrt(physical_hilbert_dim))

    # ## Define  H_tilde_1 by transforming the hamiltonian to diagonal basis 
    
    # # Create Uniform state vector 1st diagonal basis vector 
    # u_vec = (np.ones(physical_hilbert_dim)*physical_hilbert_dim)**-0.5

    # # Create set of orthogonal vectors from U to define diagonal basis vectors 
    # # This is the primitive basis   
    # primitive_vecs = []
    # for n in range(physical_hilbert_dim):
    #     primitive_vec = -(1/np.sqrt(physical_hilbert_dim))*np.copy(u_vec)
    #     primitive_vec[n] +=  1
    #     primitive_vecs.append(primitive_vec)  

    # primitive_vecs = np.array(primitive_vecs)

    # # The primitive basis is over complete we need to use a lowdin procedure 
    # overlap_matrix = primitive_vecs.T@primitive_vecs
    # eig_vals_lowdin,eig_vecs_lowdin = np.linalg.eigh(overlap_matrix)
    
    # # Check for eigenvalues that are zero they are linear dependent and can be removed 
    # # this hardcoded because the there should only be 1 excess basis vector and all other eigenvalues will be 1 
    # # linearly_independent_eigvals = eig_vals_lowdin[1:]
    # # print(linearly_independent_eigvals)

    # # Remove linear dependent vector from set of eigenvectors  
    # # Columns of eigh output are the eigenvectors 
    # linearly_independent_vecs = eig_vecs_lowdin[:,1:]

    # # Find the reduced basis set 
    # # Note: columns are basis vectors  
    # # Reduced basis written C_{m\lambda}
    # reduced_basis = primitive_vecs@linearly_independent_vecs
    
    # # Create diagonal state basis including
    # # Diagonal basis as C_{m\lambda} 
    # diagonal_basis = np.zeros((physical_hilbert_dim,physical_hilbert_dim)) 
    # diagonal_basis[:,0] = u_vec
    # diagonal_basis[:,1:] = reduced_basis

    print(hamiltonian.shape)


    # Full tensor product basis (C ⊗ C)
    C_tensor = np.kron(diagonal_basis, diagonal_basis)

    print(C_tensor.shape)

    # Full transformed V_tilde
    H_Tilde_1 = C_tensor.T @ hamiltonian @ C_tensor

    print(hamiltonian.shape)
    print(diagonal_basis.shape)

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


def V_tilde_maker(V_matrix):
    """
    Correctly constructs the full thermofield double version of a two-body potential operator.
    
    Input:
        V_matrix : np.ndarray
            Two-body potential operator, shape (d^2, d^2).
    
    Output:
        V_tilde_full : np.ndarray
            Thermofield-transformed two-body operator, shape (d^4, d^4).
    """
    dim_squared = V_matrix.shape[0]
    d = int(np.sqrt(dim_squared))
    if d * d != dim_squared:
        raise ValueError("Input matrix must be square with dimension d^2.")

    # Uniform vector
    u_vec = (np.ones(d) * d) ** -0.5

    # Primitive basis construction
    primitive_vecs = []
    for n in range(d):
        vec = -(1 / np.sqrt(d)) * np.copy(u_vec)
        vec[n] += 1
        primitive_vecs.append(vec)
    primitive_vecs = np.array(primitive_vecs)

    # Löwdin orthonormalization
    overlap_matrix = primitive_vecs.T @ primitive_vecs
    eig_vals, eig_vecs = np.linalg.eigh(overlap_matrix)
    reduced_basis = primitive_vecs @ eig_vecs[:, 1:]

    # Diagonal basis
    diagonal_basis = np.zeros((d, d))
    diagonal_basis[:, 0] = u_vec
    diagonal_basis[:, 1:] = reduced_basis

    # Full tensor product basis (C ⊗ C)
    C_tensor = np.kron(diagonal_basis, diagonal_basis)

    # Full transformed V_tilde
    H_Tilde_1 = C_tensor.T @ V_matrix @ C_tensor

        ######################## H tilde 2 and 3  
    H_Tilde_2 = np.zeros((dim_squared,dim_squared*(dim_squared)))

    for i in range(dim_squared):
        for j in range(dim_squared):
            for k in range(dim_squared):
                if j == k:     
                    H_Tilde_2[i,k + dim_squared*j] = 0 
                else: 
                    H_Tilde_2[i,k + dim_squared*j] = C_tensor[j,i]*V_matrix[j,k] 

    indices_to_keep_H2 = np.ix_([i for i in range(dim_squared*dim_squared) if (i // dim_squared) != (i % dim_squared)])[0]
    H_Tilde_2 = H_Tilde_2[:,indices_to_keep_H2]
    H_Tilde_3 = H_Tilde_2.T
    
    
    ######################## Htilde 4     
    d_tilde = np.eye((dim_squared))

    h_4 = np.kron(d_tilde,V_matrix) # The element only exist if the hamiltonian is diagonal in tilde basis  

    # Generate indices to keep (n \neq l_tilde)
    indices_to_keep_H4 = [i for i in range(dim_squared*dim_squared) if (i // dim_squared) != (i % dim_squared)]

    # Slice the matrix
    H_tilde_4 = h_4[np.ix_(indices_to_keep_H4, indices_to_keep_H4)]


    # Build H Tilde 
    H_tilde = np.zeros((d**2,d**2))
    H_tilde[:dim_squared,:dim_squared] = H_Tilde_1
    H_tilde[:dim_squared, dim_squared:] = H_Tilde_2 
    H_tilde[dim_squared:,:dim_squared] = H_Tilde_3
    H_tilde[dim_squared:,dim_squared:] = H_tilde_4

    return H_tilde