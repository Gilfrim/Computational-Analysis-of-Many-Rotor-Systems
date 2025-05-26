import numpy as np

#constructs h and v in m basis
#won't calculate anything in current state
#just here in case it's needed

states = 3
p = states

region  = int((states - 1) / 2)

h_mn = np.zeros((p, p))
for r_index, h_row in enumerate(range(-region, region + 1)):
    for c_index, h_col in enumerate(range(-region, region + 1)):
        h_mn[r_index, c_index] = h_dict.get((h_row, h_col), 0)

v_mn = np.zeros((p, p, p, p))
for zero_index, v_axis_0 in enumerate(range(-region, region + 1)):
    for one_index, v_axis_1 in enumerate(range(-region, region + 1)):
        for two_index, v_axis_2 in enumerate(range(-region, region + 1)):
            for three_index, v_axis_3 in enumerate(range(-region, region + 1)):
                v_mn[zero_index, one_index, two_index, three_index] = g * v_dict.get((v_axis_0, v_axis_1, v_axis_2, v_axis_3), 0)
