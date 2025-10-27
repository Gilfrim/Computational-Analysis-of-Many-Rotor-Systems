def free_one_body(i, j, max_m):

    # Free particle on a ring Hamiltonian in the momentum representation. 
    # K = \sum_{m=0}^{2l} (m-l)^2 |m><m|
    if i == j:
        return (i-max_m)**2
    else: 
        return 0.0

def interaction_two_body_coplanar(i1, i2, j1, j2):

    # returns <i1,i2|V|j1,j2> where V is the dipole-dipole interaction between two planar rotors oriented along the x-axis.
    # See https://arxiv.org/abs/2401.02887 for details. 
    # V = 0.75 \sum_{m1, m2} (|m1, m2> <m1+1, m2+1| + |m1, m2> <m1-1, m2-1|) - 0.25 \sum_{m1, m2} (|m1, m2> <m1-1, m2+1| + |m1, m2> <m1+1, m2-1|)
    if ((abs(i1 - j1) != 1) or (abs(i2 - j2) != 1)):
        return 0.0
    elif (i1 == j1 + 1):
        if (i2 == j2 + 1):
            return 0.75
        else:
            return -0.25
    else:
        if (i2 == j2 + 1):
            return -0.25
        else:
            return 0.75
        
def interaction_yiyj(i1, i2, j1, j2):

    # returns <i1,i2|yiyj|j1,j2> where
    # yiyj = -0.25 \sum_{m1, m2} (|m1, m2> <m1+1, m2+1| + |m1, m2> <m1-1, m2-1|) + 0.25 \sum_{m1, m2} (|m1, m2> <m1-1, m2+1| + |m1, m2> <m1+1, m2-1|)
    if ((abs(i1 - j1) != 1) or (abs(i2 - j2) != 1)):
        return 0.0
    elif (i1 - j1 == i2 - j2):
        return -0.25
    else:
        return 0.25
        
def interaction_xixj(i1, i2, j1, j2):

    # returns <i1,i2|xixj|j1,j2> where
    # xixj = 0.25 \sum_{m1, m2} (|m1, m2> <m1+1, m2+1| + |m1, m2> <m1-1, m2-1|) + 0.25 \sum_{m1, m2} (|m1, m2> <m1-1, m2+1| + |m1, m2> <m1+1, m2-1|)
    if ((abs(i1 - j1) != 1) or (abs(i2 - j2) != 1)):
        return 0.0
    else:
        return 0.25
    
def interaction_cross_terms(i1, i2, j1, j2):

    # returns <i1,i2|xiyj + yixj|j1,j2> where
    # xiyj + yixj = 0.5i \sum_{m1,m2} (|m1, m2> <m1+1, m2+1| - |m1, m2> <m1-1, m2-1|)
    if ((i1 - j1) == 1) and (i2 - j2 == 1):
        return -0.5*1J
    elif ((i1 - j1) == -1) and (i2 - j2 == -1):
        return 0.5*1J
    else: 
        return 0.0
