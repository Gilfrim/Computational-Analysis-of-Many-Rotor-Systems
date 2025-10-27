import os

import hamiltonianGenerator as hg
import numpy as np


def write_matrix_elements(m_max, fpath):

    file_dir = os.path.join(fpath, "matrix_elements_K.csv")
    d = 2 * m_max + 1
    K = np.zeros((d, d))
    with open(file_dir, "w") as f:
        f.write("Knetic Energy Matrix Elements\n")
        f.write("m1, m2, <m1|K|m2>\n")
        for i in range(d):
            for j in range(i, d):
                K[i, j] = hg.free_one_body(i, j, m_max)
                f.write(str(i) + "," + str(j) + "," + str(K[i, j]) + "\n")

    V = np.zeros((d**2, d**2))
    file_dir = os.path.join(fpath, "matrix_elements_V.csv")
    with open(file_dir, "w") as f:
        f.write("Potential 2 body Matrix Elements\n")
        f.write("m1,m2,m3,m4,<m1,m2|V|m3,m4>\n")
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        # only want to write the upper triangular parts
                        if k * d + l >= i * d + j:
                            # if hg.interaction_two_body_coplanar(i,j,k,l) != 0:
                            V[i * d + j, k * d + l] = hg.interaction_two_body_coplanar(
                                i, j, k, l
                            )
                            f.write(
                                str(i)
                                + ","
                                + str(j)
                                + ","
                                + str(k)
                                + ","
                                + str(l)
                                + ","
                                + str(V[i * d + j, k * d + l])
                                + "\n"
                            )

    return 0


if __name__ == "__main__":
    write_matrix_elements(
        5,
        r"/Users/gilfrim/Desktop/Projects/Computational_Analysis_of_Many_Rotor_Systems/z_non_periodic",
    )
