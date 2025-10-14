import csv

import numpy as np

import z_non_periodic.functions as func
from quant_rotor.models.dense.support_ham import (
    basis_m_to_p_matrix_conversion,
    write_matrix_elements,
)
from quant_rotor.models.dense.t_amplitudes_sub_class import (
    QuantumSimulation,
    SimulationParams,
    TensorData,
)

# printout settings for large matrices
np.set_printoptions(suppress = True, linewidth = 1500, threshold = 10000, precision = 12)

# expected data type of input variables and file location
expected_types = {"sites": [int, "file_path_input"],
                  "states": [int, "file_path_input"],
                  "low_states": [int, "file_path_input"],
                  "initial": [float, "file_path_input"],
                  "threshold": [float, "file_path_input"],
                  "g": [float, "file_path_input"],
                  "i_method": [int, "file_path_testing"],
                  "gap": [bool, "file_path_testing"],
                  "gap_site": [int, "file_path_testing"],
                  "transform": [bool, "file_path_testing"]}

# makes set of all possible input variables
missing_keys = set(expected_types.keys())
errors = []
values = {}

# reads through the input file for variables
file_path_input = "z_non_periodic/input.txt"
with open(file_path_input, "r") as input_file:
    for line in input_file:
        line = line.strip().lower()

        #remove empty lines
        if not line:
            continue

        #looks for comments in input file and ignores them
        line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue

        #splits lines with "=" in variable and value
        key, value = map(str.strip, line.split("=", 1))
        #dictionary keys can't have spaces so replaces with _
        key = key.replace(" ", "_")

        #if variable in input file is blank continues to next line
        if not value:
            continue

        #if the key exists removes from missing keys
        if key in expected_types:
            missing_keys.discard(key)
            expected_type = expected_types[key][0]
            try:
                values[key] = expected_type(value)
            except ValueError:
                #checks if variable type from input file matches expected
                #if they don't match adds error to list of errors
                errors.append(f"{globals()[expected_types[key][1]]}, {key}: couldn't convert to {expected_type.__name__}")

# deletes variables that are no longer needed
del file_path_input
del input_file
del line
del key
del value

# reads through the testing file for variables
file_path_testing = "z_non_periodic/testing.txt"
with open(file_path_testing, "r") as testing_file:
    for line_testing in testing_file:
        line_testing = line_testing.strip()

        #checks for empty lines
        if not line_testing:
            continue

        #checks for comments
        line_testing = line_testing.split("#", 1)[0].strip()
        #checks for variables
        if "=" not in line_testing:
           continue

        key_testing, value_testing = map(str.strip, line_testing.split("=", 1))
        key_testing = key_testing.replace(" ", "_")

        #converts true and false strings to bool
        #works for both True, False and true, false
        if key_testing == "gap" or key_testing == "transform":
            value_testing = value_testing.capitalize()
            value_testing = func.str_to_bool(value_testing)

        #checks for empty variable values
        if value_testing == "":
            continue

        #checks for variables, removes from missing keys if it exists
        if key_testing in expected_types:
            missing_keys.discard(key_testing)
            expected_type = expected_types[key_testing][0]
            try:
                values[key_testing] = expected_type(value_testing)
            except ValueError:
                #checks if variable type from input file matches expected
                #if they don't match adds error to list of errors
                errors.append(f"{globals()[expected_types[key_testing][1]]}, {key_testing}: couldn't convert to {expected_type.__name__}")

# delete variables that are no longer needed
del file_path_testing
del testing_file
del line_testing
del key_testing
del value_testing

# adds missing keys to error list
if missing_keys:
    errors.append(f"Missing required values:")
    for missing in missing_keys:
        errors.append(f"{expected_types[missing][1]}: {missing}")

del expected_types
del missing_keys

# raise list of errors
if errors:
    #raise ValueError("Invalid input detected:\n" + "\n".join(errors))
    raise ValueError(f"Invalid input:\n{"\n".join(errors)}")

del errors

# makes variables from text files
sites = values["sites"]
states = values["states"]
low_states = values["low_states"]
initial = values["initial"]
threshold = values["threshold"]
g = values["g"]
i_method = values["i_method"]
gap = values["gap"]
gap_site = values["gap_site"] - 1
transform = values["transform"]

del values

# state variables
# could just use p, i, a
# makes checking einsums and such a bit easier
p = q = r = s = states
i = j = k = l = low_states
a = b = c = d = p - i

# checks for i and a sharing same state level
if i %2 == 0 or a %2 != 0:
    raise ValueError("Overlap between low and high stats, will cause divide by zero in denominator update")

m_max = abs(func.p_to_m(states - 1))
m_max_shaeer = 5

# for states > 11 would need to have shaeers code generate csv files with m_max > 5
# gets h values from csv files made with Shaeer's code and puts them into a dictionary
h_dict = dict()
with open(r"z_non_periodic/matrix_elements_K.csv", mode = "r",newline = "") as csvfile_h:
    reader_h = csv.reader(csvfile_h, delimiter = ",")
    next(reader_h)
    next(reader_h)
    for row_h in reader_h:
        if float(row_h[2]) != 0.0:
            h_dict[((int(row_h[0]) - m_max_shaeer), (int(row_h[1]) - m_max_shaeer))] = float(row_h[2])

# creates h term of size (p, p) so that it can be sliced
h_full = np.zeros((p, p))
for h_row in range(p):
    for h_col in range(p):
        h_full[h_row, h_col] = h_dict.get((func.p_to_m(h_row), func.p_to_m(h_col)), 0)

# h_dict is no longer needed due to h_full
del h_dict

# gets v values from csv file made Shaeer's code and puts them into a dictionary
v_dict = dict()
with open(r"z_non_periodic/matrix_elements_V.csv", mode = "r",newline = "") as csvfile_v:
    reader_v = csv.reader(csvfile_v, delimiter = ",")
    next(reader_v)
    next(reader_v)
    for row_v in reader_v:
        if float(row_v[4]) != 0.0:
            v_dict[((int(row_v[0]) - m_max_shaeer), (int(row_v[1]) - m_max_shaeer), (int(row_v[2]) - m_max_shaeer), (int(row_v[3]) - m_max_shaeer))] = float(row_v[4])
            if int(row_v[0]) != int(row_v[2]) and int(row_v[1]) != int(row_v[3]):
                v_dict[((int(row_v[2]) - m_max_shaeer), (int(row_v[3]) - m_max_shaeer), (int(row_v[0]) - m_max_shaeer), (int(row_v[1]) - m_max_shaeer))] = float(row_v[4])

# creates maximum size v term (p, p, p, p) so that it can be sliced
v_full = np.zeros((p, p, p, p))
for v_axis_0 in range(p):
    for v_axis_1 in range(p):
        for v_axis_2 in range(p):
            for v_axis_3 in range(p):
                v_full[v_axis_0, v_axis_1, v_axis_2, v_axis_3] = g * v_dict.get((func.p_to_m(v_axis_0), func.p_to_m(v_axis_1), func.p_to_m(v_axis_2), func.p_to_m(v_axis_3)), 0)

# v_dict no longer needed due to v_full
del v_dict

# t1 and t2 amplitude tensors
t_a_i_tensor = np.full((sites, a, i), 0.0)
t_ab_ij_tensor = np.full((sites, sites, a, b, i, j), 0.0)

# t_a_i_tensor = np.random.rand(sites, a, i)
# t_ab_ij_tensor = np.random.rand(sites, sites, a, a, i, i)

# eigenvalues from h for update
epsilon = np.diag(h_full)

def A_term(a_upper: int, a_lower: int, a_site: int)->np.array:
    """Generates the A commutator matrix based on upper and lower index and site given"""
    #are upper and lower indices needed if always shape A^a_p

    #horizontally combines the t_a_i amplitudes from a_site with identity matrix of size a
    A = np.hstack((-t_a_i_tensor[a_site], np.identity(a_upper)))
    return A

def B_term(b_upper: int, b_lower: int, b_site)->np.array:
    """Generates the B commutator matrix based on upper and lower index and site given"""
    #are indices needed (see A)

    #vertically combines identity matrix size i with t_a_i amplitudes from site_b
    B = np.vstack((np.identity(b_lower), t_a_i_tensor[b_site]))
    return B

# unitary transformation test
# same as transformation test.py
if transform:
    h_off = h_full.copy()

    for p_off in range(states):
        for q_off in range(states):
            if p_off != q_off:
                h_off[p_off, q_off] = 0.1 / abs(p_off - q_off)

    _, U = np.linalg.eigh(h_off)
    h_full = U.T @ h_full @ U

    # noinspection SpellCheckingInspection
    v_full = np.einsum("ip, jr, prqs, qk, sl->ijkl", U.T, U.T, v_full, U, U)

def h_term(h_upper: int, h_lower: int, h_site: int)->np.array:
    """Slices h_full based on given indices"""

    a_h_shift = [i if a_check == a else 0 for a_check in (h_upper, h_lower)]
    return h_full[a_h_shift[0]:h_upper + a_h_shift[0], a_h_shift[1]:h_lower + a_h_shift[1]]

def v_term(v_upper_1: int, v_upper_2: int, v_lower_1: int, v_lower_2: int, v_site_1: int, v_site_2: int)->np.array:
    """Slices v_full based on given indices and sites"""

    #gap is testing parameter to check if example 5 site system gap between site 2 and 3 is the same as energy of 2 rotors + energy of 3 rotors
    if gap == True and (v_site_1 == gap_site and (v_site_2 == gap_site + 1) or (v_site_1 == gap_site + 1 and v_site_2 == gap_site)):
        return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))

    #checks for nearest neighbour
    #should also try and implement in einsum loops to skip some x0 calculations
    if abs(v_site_1 - v_site_2) == 1:
        #if the index = a shifts the array slicing to (i + 1) to p
        a_v_shift = [i if a_check == a else 0 for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)]
        return v_full[a_v_shift[0]:v_upper_1 + a_v_shift[0], a_v_shift[1]:v_upper_2 + a_v_shift[1], a_v_shift[2]:v_lower_1 + a_v_shift[2], a_v_shift[3]:v_lower_2 + a_v_shift[3]]
    else:
        return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))

def t_term(t_upper_1: int, t_upper_2: int, t_lower_1: int, t_lower_2: int, t_site_1: int, t_site_2: int)->np.array:
    """Slices t_ab_ij to give t amplitudes for 2 sites"""
    return t_ab_ij_tensor[t_site_1, t_site_2]

# "# noinspection SpellCheckingInspection" are for pycharm to ignore the einsum strings it thinks are typos  i.e. "plcd"
# not needed, I found the typo errors annoying

def residual_single(x_s:int)->np.array:
    """Calculates R^{a}_{i}(x) singles equation"""
    R_single = np.zeros((a, i))

    R_single += np.einsum("ap, pq, qi->ai", A_term(a, p, x_s), h_term(p, q, x_s), B_term(q, i, x_s))
    for z_s in range(sites):
        if z_s != x_s:
            if i_method >= 1:
                # noinspection SpellCheckingInspection
                R_single += np.einsum("ap, plcd, cdil->ai", A_term(a, p, x_s), v_term(p, l, c, d, x_s, z_s), t_term(c, d, i, l, x_s, z_s))
            # noinspection SpellCheckingInspection
            R_single += np.einsum("ap, plqs, qi, sl->ai", A_term(a, p, x_s), v_term(p, l ,q, s, x_s, z_s), B_term(q, i, x_s), B_term(s, l, z_s))

    return R_single

def residual_double_sym(x_ds:int, y_ds:int)->np.array:
    """Calculates Rs^{ab}_{ij}(x < y) symmetric doubles equation"""
    R_double_symmetric = np.zeros((a, b, i, j))

    if i_method >= 1:
        # noinspection SpellCheckingInspection
        R_double_symmetric += np.einsum("ap, bq, pqrs, ri, sj->abij", A_term(a, p, x_ds), A_term(b, q, y_ds), v_term(p, q, r, s, x_ds, y_ds), B_term(r, i, x_ds), B_term(s, j, y_ds))
        if i_method >= 2:
            # noinspection SpellCheckingInspection
            R_double_symmetric += np.einsum("ap, bq, pqcd, cdij->abij", A_term(a, p, x_ds), A_term(b, q, x_ds), v_term(p, q, c, d, x_ds, y_ds), t_term(c, d, i, j, x_ds, y_ds))
            # noinspection SpellCheckingInspection
            R_double_symmetric -= np.einsum("abkl, klpq, pi, qj->abij", t_term(a, b, k, l, x_ds, y_ds), v_term(k, l, p, q, x_ds, y_ds), B_term(p, i, x_ds), B_term(q, j, y_ds))
            if i_method == 3:
                # noinspection SpellCheckingInspection
                R_double_symmetric -= np.einsum("abkl, klcd, cdij->abij", t_term(a, b, k, l, x_ds, y_ds), v_term(k, l, c, d, x_ds, y_ds), t_term(c, d, i, j, x_ds, y_ds))
                if sites >= 4:
                    for z_ds in range(sites):
                        for w_ds in range(sites):
                            if z_ds not in {x_ds, y_ds} and w_ds not in {x_ds, y_ds} and z_ds != w_ds:
                                # noinspection SpellCheckingInspection
                                R_double_symmetric += np.einsum("klcd, acik, bdjl->abij", v_term(k, l, c, d, z_ds, w_ds), t_term(a, c, i, k, x_ds, z_ds), t_term(b, d, j, l, y_ds, w_ds))

    return R_double_symmetric

def residual_double_non_sym_1(x_dns_1:int, y_dns_1:int)->np.array:
    """Calculates Rn^{ab}_{ij}(x, y) non-symmetric doubles equation"""
    R_double_non_symmetric_1 = np.zeros((a, b, i, j))

    if i_method >= 1:
        # noinspection SpellCheckingInspection
        R_double_non_symmetric_1 += np.einsum("ap, pc, cbij->abij", A_term(a, p, x_dns_1), h_term(p, c, x_dns_1), t_term(c, b, i, j, x_dns_1, y_dns_1))
        # noinspection SpellCheckingInspection
        R_double_non_symmetric_1 -= np.einsum("abkj, kp, pi->abij", t_term(a, b, k, j, x_dns_1, y_dns_1), h_term(k, p, x_dns_1), B_term(p, i, x_dns_1))

        if i_method >= 2:
            for z_dns_1 in range(sites):
                if z_dns_1 != x_dns_1 and z_dns_1 != y_dns_1:
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_1 += np.einsum("acik, krcs, br, sj->abij", t_term(a, c, i, k, x_dns_1, z_dns_1), v_term(k, r, c, s, z_dns_1, y_dns_1), A_term(b, r, y_dns_1), B_term(s, j, y_dns_1))
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_1 += np.einsum("bq, qlds, adij, sl->abij", A_term(b, q, y_dns_1), v_term(q, l, d, s, y_dns_1, z_dns_1), t_term(a, d, i, j, x_dns_1, y_dns_1), B_term(s, l, z_dns_1))
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_1 -= np.einsum("abkj, lkrp, pi, rl->abij", t_term(a, b, k, j, x_dns_1, y_dns_1), v_term(l, k, r, p, z_dns_1, x_dns_1), B_term(p, i, x_dns_1), B_term(r, l, z_dns_1))

    return R_double_non_symmetric_1

def residual_double_non_sym_2(x_dns_2:int, y_dns_2:int)->np.array:
    """Calculates Rn^{ba}_{ji}(y, x) ai -> jb permutation non-symmetric doubles equation"""
    R_double_non_symmetric_2 = np.zeros((b, a, j, i))

    if i_method >= 1:
        # noinspection SpellCheckingInspection
        R_double_non_symmetric_2 += np.einsum("bp, pc, caji->baji", A_term(b, p, y_dns_2), h_term(p, c, y_dns_2), t_term(c, a, j, i, y_dns_2, x_dns_2))
        # noinspection SpellCheckingInspection
        R_double_non_symmetric_2 -= np.einsum("baki, kp, pj->baji", t_term(b, a, k, i, y_dns_2, x_dns_2), h_term(k, p, y_dns_2), B_term(p, j, y_dns_2))

        if i_method >= 2:
            for z_dns_2 in range(sites):
                if z_dns_2 != x_dns_2 and z_dns_2 != y_dns_2:
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_2 += np.einsum("bcjk, krcs, ar, si->baji", t_term(b, c, j, k, y_dns_2, z_dns_2), v_term(k, r, c, s, z_dns_2, x_dns_2), A_term(a, r, x_dns_2), B_term(s, i, x_dns_2))
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_2 += np.einsum("aq, qlds, bdji, sl->baji", A_term(a, q, x_dns_2), v_term(q, l, d, s, x_dns_2, z_dns_2), t_term(b, d, j, i, y_dns_2, x_dns_2), B_term(s, l ,z_dns_2))
                    # noinspection SpellCheckingInspection
                    R_double_non_symmetric_2 -= np.einsum("baki, lkrp, pj, rl->baji", t_term(b, a, k, i, y_dns_2, x_dns_2), v_term(l, k, r, p, z_dns_2, y_dns_2), B_term(p, j, y_dns_2), B_term(r, l ,z_dns_2))

    return R_double_non_symmetric_2

def residual_double_total(x_d:int, y_d:int)->np.array:
    """Calculates Rt^{ab}_{ij}(x < y) = Rs^{ab}_{ij}(x < y) + Rn^{ab}_{ij}(x < y) + Rn^{ba}_{ji}(y < x)total doubles equation"""

    return residual_double_sym(x_d, y_d) + residual_double_non_sym_1(x_d, y_d) + residual_double_non_sym_2(x_d, y_d)

def update_one(r_1_value:np.array)->np.array:
    """Calculates the iterative update value for t_1 amplitudes"""
    #the negative from the equation is in the while loop, approx line 459
    update_1 = np.zeros((a, i))

    for u_a_1 in range(a):
        for u_i_1 in range(i):
            update_1[u_a_1, u_i_1] = 1 / (epsilon[u_a_1 + i] - epsilon[u_i_1])

    return np.multiply(update_1, r_1_value)

def update_two(r_2_value:np.array)->np.array:
    """Calculates the iterative update value for t_2 amplitudes"""
    #the negative from the equation is in the while loop, approx line 462
    update_2 = np.zeros((a, b, i, j))

    for u_a_2 in range(a):
        for u_b_2 in range(b):
            for u_i_2 in range(i):
                for u_j_2 in range(j):
                    update_2[u_a_2, u_b_2, u_i_2, u_j_2] = 1 / (epsilon[u_a_2 + i] + epsilon[u_b_2 + j] - epsilon[u_i_2] - epsilon[u_j_2])

    return np.multiply(update_2, r_2_value)

def t_non_periodic():

    t_a_i_tensor_new = np.full((sites, a, i), 0, dtype=complex)
    t_ab_ij_tensor_new = np.full((sites, sites, a, a, i, i), 0, dtype=complex)

    # t_a_i_tensor_new = t_a_i_tensor
    # t_ab_ij_tensor_new = t_ab_ij_tensor

    params = SimulationParams(
    a=a,
    i=i,
    p=p,  # These can be the same as `a + i` or chosen independently
    site=sites,
    state=states,
    i_method=i_method,
    gap=gap,
    gap_site=gap_site,
    epsilon=epsilon,
    periodic=False
    )

    tensors = TensorData(
        t_a_i_tensor=t_a_i_tensor_new,
        t_ab_ij_tensor=t_ab_ij_tensor_new,
        h_full=h_full,
        v_full=v_full,
    )

    qs = QuantumSimulation(params, tensors)

    single_new = np.zeros((sites, a, i), dtype=complex)
    double_new = np.zeros((sites, sites, a, a, i, i), dtype=complex)

    # iteration counter
    iteration = 0
    # single and double residual initialization
    single = np.zeros((sites, a, i))
    double = np.zeros((sites, sites, a, b, i ,j))

    # print("Old Double:")

    # x_d, y_d = 0, 1

    # print("A_term", np.allclose(qs.A_term(a, x_d), A_term(a, p, x_d)))
    # print("A_term", np.allclose(qs.A_term(a, y_d), A_term(a, p, y_d)))

    # print("B_term", np.allclose(qs.B_term(i, x_d), B_term(r, i, x_d)))
    # print("B_term", np.allclose(qs.B_term(i, y_d), B_term(r, i, y_d)))

    # print(
    #     "v_term",
    #     np.allclose(qs.v_term(p, p, p, p, x_d, y_d), v_term(p, p, p, p, x_d, y_d)),
    # )
    # print(
    #     "v_term",
    #     np.allclose(qs.v_term(p, p, a, a, x_d, y_d), v_term(p, p, a, a, x_d, y_d)),
    # )
    # print(
    #     "v_term",
    #     np.allclose(qs.v_term(i, i, p, p, x_d, y_d), v_term(i, i, p, p, x_d, y_d)),
    # )
    # print(
    #     "v_term",
    #     np.allclose(qs.v_term(i, i, a, a, x_d, y_d), v_term(i, i, a, a, x_d, y_d)),
    # )

    # print(
    #     "t_term",
    #     np.allclose(qs.t_term(x_d, y_d), t_term(a, b, k, l, x_d, y_d)),
    # )

    # used to calculate change in energy between iterations
    previous_energy = 0
    # runs until residual < threshold
    # should maybe also add if iterative > value ask to continue prompt or break

    while True:
        energy = 0
        energy_new = 0

        for x_site in range(sites):
            single_new[x_site] = qs.residual_single(x_site)
            single[x_site] = residual_single(x_site)

            for y_site in range(sites):
                if x_site < y_site:
                    double_new[x_site, y_site] = qs.residual_double_total(
                        x_site, y_site
                    )
                    double[x_site, y_site] = residual_double_total(x_site, y_site)

        for x_site in range(sites):
            single[x_site] = residual_single(x_site)
            for y_site in range(sites):
                if x_site < y_site:
                    double[x_site, y_site] = residual_double_total(x_site, y_site)

        # prints largest magnitude value from both residual equations
        # nice to see while to get a feel for how long it might take to run
        # print(f"1 max: {np.max(np.abs(single))}")
        # print(f"2 max: {np.max(np.abs(double))}\n")

        # checks if both residuals are all < threshold
        # ends program

        for site_u_1 in range(sites):
            tensors.t_a_i_tensor[site_u_1] -= qs.update_one(single_new[site_u_1])
            for site_u_2 in range(sites):
                if site_u_1 < site_u_2:
                    tensors.t_ab_ij_tensor[site_u_1, site_u_2] -= qs.update_two(
                        double_new[site_u_1, site_u_2]
                    )

        # calculates update values for residuals
        for site_u_1 in range(sites):
            t_a_i_tensor[site_u_1] -= update_one(single[site_u_1])
            for site_u_2 in range(sites):
                if site_u_1 < site_u_2:
                    t_ab_ij_tensor[site_u_1, site_u_2] -= update_two(
                        double[site_u_1, site_u_2]
                    )
                    # t_ab_ij_tensor[site_u_2, site_u_1] -= update_two(double[site_u_1, site_u_2])

        # energy calculations
        for site_x in range(sites):
            energy += np.einsum("ip, pi->", h_term(i, p, site_x), B_term(p, i, site_x))
            for site_y in range(sites):
                if site_x < site_y:
                    # noinspection SpellCheckingInspection
                    energy += np.einsum(
                        "ijab, abij->",
                        v_term(i, j, a, b, site_x, site_y),
                        t_term(a, b, i, j, site_x, site_y),
                    )
                    # noinspection SpellCheckingInspection
                    energy += np.einsum(
                        "ijpq, pi, qj->",
                        v_term(i, j, p, q, site_x, site_y),
                        B_term(p, i, site_x),
                        B_term(q, j, site_y),
                    )

        if np.all(abs(single) <= threshold) and np.all(abs(double) <= threshold):
            break

        # checks for program diverging to infinity
        if (
            np.isnan(single).any()
            or np.isnan(double).any()
            or np.isinf(single).any()
            or np.isinf(double).any()
        ):
            raise ValueError("Diverges (inf or nan)")

    print(energy)
    return (
        np.max(np.abs(single)),
        np.max(np.abs(double)),
        energy,
        t_a_i_tensor,
        t_ab_ij_tensor,
    )
