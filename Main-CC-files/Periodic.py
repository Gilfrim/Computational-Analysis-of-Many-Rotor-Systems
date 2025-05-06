import numpy as np
import csv
import functions as func
import writeMatrixElements as wME

#printout settings for large matrices
np.set_printoptions(suppress = True, linewidth = 1500, threshold = 10000, precision = 12)

#location of folder for code and such
file_path_folder = r"/Users/gilfrim/Desktop/QuantumChemistryCoop/Main-CC-files"
file_path_K = file_path_folder + r"/matrix_elements_K.csv"
file_path_V = file_path_folder + r"/matrix_elements_V.csv"

#expected data type of input variables and file location
expected_types = {"sites": [int, "file_path_input"],
                  "states": [int, "file_path_input"],
                  "low_states": [int, "file_path_input"],
                  "initial": [float, "file_path_input"],
                  "threshold": [float, "file_path_input"],
                  "g": [float, "file_path_input"],
                  "i_method": [int, "file_path_testing"],
                  "HF": [bool, "file_path_testing"],
                  "start_point": [str, "file_path_testing"]}

#makes set of all possible input variables
missing_keys = set(expected_types.keys())
errors = []
values = {}

#goes through input.txt to find variables
file_path_input = "input.txt"
with open(file_path_input, "r") as input_file:
    for line in input_file:
        line = line.strip().lower()

        #skip empty lines
        if not line:
            continue

        #looks for comments in input file and ignores them
        line = line.split("#", 1)[0].strip()

        #only need lines with containing =
        if "=" not in line:
            continue

        #splits lines with "=" in variable and value
        key, value = map(str.strip, line.split("=", 1))
        #dictionary keys can't have spaces so replaces with _
        key = key.replace(" ", "_")

        #if variable in input file is blank continues to next line
        if value == "":
            continue

        #if the variable is blank it won't be removed from missing keys
        if key in expected_types:
            missing_keys.discard(key)
            expected_type = expected_types[key][0]
            try:
                values[key] = expected_type(value)
            except ValueError:
                #checks if variable type from input file matches expected
                #if they don't match adds error to list of errors
                errors.append(f"{globals()[expected_types[key][1]]}, {key}: couldn't convert to {expected_type.__name__}")

del file_path_input
del input_file
del line
del key
del value

#goes through testing.txt for variables
file_path_testing = "testing.txt"
with open(file_path_testing, "r") as testing_file:
    for line_testing in testing_file:
        line_testing = line_testing.strip()

        #skip empty lines
        if not line_testing:
            continue

        #removes comments
        line_testing = line_testing.split("#", 1)[0].strip()
        if "=" not in line_testing:
           continue

        #splits into variable and value
        key_testing, value_testing = map(str.strip, line_testing.split("=", 1))
        key_testing = key_testing.replace(" ", "_")

        #converts transform from string to bool
        if key_testing == "HF":
            value_testing = value_testing.capitalize()
            value_testing = func.str_to_bool(value_testing)

        #if line is empty skip
        if value_testing == "":
            continue

        #checks that all keys exist
        if key_testing in expected_types:
            missing_keys.discard(key_testing)
            expected_type = expected_types[key_testing][0]
            try:
                values[key_testing] = expected_type(value_testing)
            except ValueError:
                #checks if variable type from input file matches expected
                #if they don't match adds error to list of errors
                errors.append(f"{globals()[expected_types[key_testing][1]]}, {key_testing}: couldn't convert to {expected_type.__name__}")

del file_path_testing
del testing_file
del line_testing
del key_testing
del value_testing

#adds missing keys to error list
if missing_keys:
    #errors.append(f"Missing required values:\n {missing_keys}")
    errors.append(f"Missing required values:")
    for missing in missing_keys:
        errors.append(f"{expected_types[missing][1]}: {missing}")

del expected_types
del missing_keys

#raise list of errors
if errors:
    #raise ValueError("Invalid input detected:\n" + "\n".join(errors))
    raise ValueError(f"Invalid input:\n{"\n".join(errors)}")

del errors

#variables values from dictionary
sites = values["sites"]
states = values["states"]
low_states = values["low_states"]
initial = values["initial"]
threshold = values["threshold"]
g = values["g"]
i_method = values["i_method"]
HF = values["HF"]
start_point = values["start_point"]

if start_point != "cos" and start_point != "sin":
    raise ValueError("Starting point must be either cos or sin")

del values

#site labels
#only p, i, a are needed but these extra ones make the einsums and h v a b terms more readable
p = q = r = s = states
i = j = k = l = low_states
a = b = c = d = p - i

#if a == i and (a != 1 and i != 1):
if i %2 == 0:
    raise ValueError("Overlap between low and high stats, will cause divide by zero in denominator update")

#checks for valid i method value
if i_method not in [0, 1, 2, 3]:
    raise ValueError("I method value outside range")

#maximum m basis value
m_max = abs(func.p_to_m(states - 1))

#checks for the max generated size of h and v values
with open(file_path_K, mode = "r",newline = "") as check:
    reader_check = csv.reader(check)
    for row_check in reader_check:
        pass
    max_check = int(row_check[0]) + 1

#if the max basis value is less than the amount in csv calls shaeer's code to generate larger basis
if max_check < states:
    wME.write_matrix_elements(m_max, file_path_folder)

#shaeer's code generates 0 to 2 * L instead of -L to L so it needs to be shifted over for correct m indexing
max_shift = func.p_to_m(max_check - 1)

#gets h values from csv files made with Shaeer's code and puts them into a dictionary
h_dict = dict()

#values for h term
with open(file_path_K, mode = "r",newline = "") as csvfile_h:
    reader_h = csv.reader(csvfile_h, delimiter = ",")
    next(reader_h)
    next(reader_h)
    for row_h in reader_h:
        if float(row_h[2]) != 0.0:
            h_dict[((int(row_h[0]) - max_shift), (int(row_h[1]) - max_shift))] = float(row_h[2])

#creates h term of size (p, p) so that it can be sliced
h_full = np.zeros((p, p))

#goes through h dict to populate max size h term
for h_row in range(p):
    for h_col in range(p):
        h_full[h_row, h_col] = h_dict.get((func.p_to_m(h_row), func.p_to_m(h_col)), 0)

#h_dict is no longer needed due to h_full
del h_dict

#dictionary for v values from shaeer's code
v_dict = dict()
with open(file_path_V, mode = "r",newline = "") as csvfile_v:
    reader_v = csv.reader(csvfile_v, delimiter = ",")
    next(reader_v)
    next(reader_v)
    for row_v in reader_v:
        if float(row_v[4]) != 0.0:
            v_dict[((int(row_v[0]) - max_shift), (int(row_v[1]) - max_shift), (int(row_v[2]) - max_shift), (int(row_v[3]) - max_shift))] = float(row_v[4])
            #shaeer's code only does upper triangular values
            if int(row_v[0]) != int(row_v[2]) and int(row_v[1]) != int(row_v[3]):
                v_dict[((int(row_v[2]) - max_shift), (int(row_v[3]) - max_shift), (int(row_v[0]) - max_shift), (int(row_v[1]) - max_shift))] = float(row_v[4])

#creates maximum size v term (p, p, p, p) so that it can be sliced
v_full = np.zeros((p, p, p, p))

#populates max size v term
for v_axis_0 in range(p):
    for v_axis_1 in range(p):
        for v_axis_2 in range(p):
            for v_axis_3 in range(p):
                v_full[v_axis_0, v_axis_1, v_axis_2, v_axis_3] = g * v_dict.get((func.p_to_m(v_axis_0), func.p_to_m(v_axis_1), func.p_to_m(v_axis_2), func.p_to_m(v_axis_3)), 0)

#v_dict no longer needed due to v_full
del v_dict

#starting point using sin needs complex values
if HF and start_point == "sin":
    t_a_i_tensor = np.full((sites, a, i), initial, dtype=complex)
    t_ab_ij_tensor = np.full((sites, sites, a, b, i, j), initial, dtype=complex)
#otherwise use real matrices
else:
    t_a_i_tensor = np.full((sites, a, i), initial)
    t_ab_ij_tensor = np.full((sites, sites, a, b, i, j), initial)

for set_zero in range(sites):
    t_ab_ij_tensor[set_zero, set_zero, :, :, :, :] = 0

if HF:
    #crit point negative is just for testing should change
    crit_point = -0.5
    # Initialize the density matrix based on the starting point and the matrix form
    if g > crit_point:
        # ⟨m|cosφ|n⟩ = 0.5 * δ_{m, n+1} + 0.5 * δ_{m, n-1}
        # ⟨m|sinφ|n⟩ = 1/(2i) * δ_{m, n+1} + 1/(2i) * δ_{m, n-1}
        if start_point == "sin":
            matrix = np.zeros((states, states), dtype = complex)
        else:
            matrix = np.zeros((states, states))

        #uses real values for cos and complex for sin
        t = {"cos": 0.5, "sin": 1 / 2j}[start_point]

        #adds elements to trig matrix
        for off_diag in range(states - 1):
            matrix[off_diag, off_diag + 1] = t
            matrix[off_diag + 1, off_diag] = np.conj(t)


        # ⟨m|cos^2φ|n⟩ = 0.5 * δ_{m, n} + 0.5 * (δ_{m, n+2} + δ_{m, n-2})
        # ⟨m|cos^2φ|n⟩ = 0.5 * δ_{m, n} - 0.5 * (δ_{m, n+2} + δ_{m, n-2})
        I = np.identity(states)
        cos_2phi = np.zeros((states, states))

        for off in range(states - 2):
            cos_2phi[off, off + 2] = 1
            cos_2phi[off + 2, off] = 1

        # cos^2φ = (1 + cos(2φ))/ 2
        # sin^2φ = (1 - cos(2φ))/ 2
        if start_point == "cos":
            matrix_squared = 0.5 * (I + cos_2phi)
        elif start_point == "sin":
            matrix_squared = 0.5 * (I - cos_2phi)

    else:
        #uses h as starting point if g < crit point
        matrix = h_full

    #stuff I never got to
    #mix_factor = -0.1
    #h_full += mix_factor * matrix

    vals, vecs = np.linalg.eigh(matrix)

    #occupied orbitals
    U_occ = vecs[:, :i]
    density = U_occ @ U_occ.conj().T

    iteration_density = 0
    #hartree fock iterative calculations
    while True:
        iteration_density += 1
        #could use 2 * np.einsum("mknl, lk ->mn", v_full, density) instead, gives same thing
        # noinspection SpellCheckingInspection
        fock = h_full + np.einsum("mknl, lk ->mn", v_full, density) + np.einsum("mknl, mn ->lk", v_full, density)
        #makes sure fock is hermitian
        fock = 0.5 * fock + 0.5 * fock.conj().T

        fock_val, fock_vec = np.linalg.eigh(fock)

        #checks that fock eigenvectors are unitary
        if not np.allclose(fock_vec.conj().T @ fock_vec, np.identity(states)):
            raise ValueError("Fock vectors not unitary")

        print(f"Iteration density: {iteration_density}")

        #calculates initial fock occupied orbital
        if iteration_density == 1:
            f_occ = fock_vec[:, :i]
            f_occ_zero = f_occ.copy()
        else:
            #overlap integral
            overlap = abs(np.einsum("ka, kl->la", f_occ_zero, fock_vec))
            overlap_index = np.argmax(abs(overlap))

            #changes largest overlap eigenvector and eigenvalue to be first
            if overlap_index != 0:
                #switches the largest overlap vector to 0th position
                fock_val[[0, overlap_index]] = fock_val[[overlap_index, 0]]
                fock_vec[:, [0, overlap_index]] = fock_vec[:, [overlap_index, 0]]

            f_occ = fock_vec[:, :i]

        #calculates new density matrix
        density_update = np.einsum("ki, li ->kl", f_occ, f_occ)
        max_change = np.max(np.abs(density - density_update))
        print(f"max change: {max_change}")

        #how much the density matrix gets updated
        mix_param = 0.5
        density = mix_param * density + (1 - mix_param) * density_update

        if g > crit_point:
            #calculates the expectation value and variance of cosφ
            #haven't tested for sinφ, might need some changes
            expval = np.trace(matrix @ density)
            expval_squared = np.trace(matrix_squared @ density)

            print(f"⟨{start_point}⟩ = {np.real(expval)}")

            variance = expval_squared - expval ** 2

            print(f"Var({start_point}) = {np.real(variance)}\n")

        #stopping condition for fock matrix updates
        if max_change < 1e-10 or iteration_density == 250:
            break

    #once the density matrix has converged calculate fock lat time
    fock_final = h_full + 2 * np.einsum("mknl, lk ->mn", v_full, density)

    fock_final = 0.5 * fock_final + 0.5 * fock_final.conj().T
    fock_final_val, fock_final_vec = np.linalg.eigh(fock_final)

    #overlap integrals
    overlap_final = abs(np.einsum("ka, kl->la", f_occ_zero, fock_final_vec))
    overlap_index_final = np.argmax(abs(overlap_final))

    if overlap_index_final != 0:
        fock_final_val[[0, overlap_index_final]] = fock_final_val[[overlap_index_final, 0]]
        fock_final_vec[:, [0, overlap_index_final]] = fock_final_vec[:, [overlap_index_final, 0]]

    f_final_occ = fock_final_vec[:, :i]

    #check eigenvectors unitary
    if not np.allclose(fock_final_vec.conj().T @ fock_final_vec, np.identity(states)):
        raise ValueError("Fock final vectors not unitary")

    #checks that the difference between density and 2nd last density are the same
    density_final = np.einsum("pi, qi ->pq", f_final_occ, f_final_occ)
    if not np.allclose(density_final,density):
        print(density)
        print(density_final)
        raise ValueError("Density problem")

    #various tests to check that basis transformations are working
    #h_pq = h_full.copy()
    #v_pqrs = v_full.copy()

    #Need these
    #transformation to fock basis
    h_full = fock_final_vec.conj().T @ h_full @ fock_final_vec
    v_full = np.einsum("pi, qj, pqrs, rk, sl->ijkl", fock_final_vec, fock_final_vec, v_full, fock_final_vec, fock_final_vec)

    #Should maybe make these some functions that you call instead
    #more test stuff
    #u_pj = fock_final_vec

    #i_pq = np.einsum("pj, qj ->pq", u_pj, u_pj)
    #print(f"U pj U qj unitary test: {np.allclose(i_pq, np.identity(states))}")

    #i_ij = np.einsum("pj, pi ->ij", u_pj, u_pj)
    #print(f"U pj U pi unitary test: {np.allclose(i_ij, np.identity(states))}")

    #d_pq = density_final.copy()

    #d_ij = np.einsum("pi, pq, qj->ij",u_pj, d_pq, u_pj)
    #print(f"D ij\n{d_ij}")

    #d_ij_check = np.zeros((states, states))
    #d_ij_check[0, 0] = 1
    #print(f"D ij test: {np.allclose(d_ij, d_ij_check)}")

    #f_pq_0 = fock_final

    #f_ij_0 = np.einsum("pi, pq, qj ->ij", u_pj, f_pq_0, u_pj)

    #f_ij_0_check = np.diag(fock_final_val)
    #print(f"F ij 0 test: {np.allclose(f_ij_0, f_ij_0_check)}")

    #f_pq_1 = h_pq + 2 * np.einsum("prqs, sr ->pq", v_pqrs, d_pq)
    #print(f"F pq 0 and 1 check: {np.allclose(f_pq_0, f_pq_1)}")

    #v_piqj = np.einsum("prqs, ri, sj -> piqj", v_pqrs, u_pj, u_pj)

    #f_pq_2 = h_pq + 2 * np.einsum("piqj, ij ->pq", v_piqj, d_ij)
    #print(f"F pq 0 and 2 check: {np.allclose(f_pq_0, f_pq_2)}")

    #h_ij = h_full.copy()

    #f_ij_1 = h_ij + 2 * np.einsum("piqj, qp", v_piqj, d_pq)
    #print(f"F ij 0 and 1 check: {np.allclose(f_ij_0, f_ij_1)}")

    #v_ijkl = v_full.copy()

    #f_ij_2 = h_ij + 2 * np.einsum("ikjl, lk ->ij", v_ijkl, d_ij)
    #print(f"F ij 0 and 2 check: {np.allclose(f_ij_0, f_ij_2)}")

    #v_aux = 0.5 * (v_aux + v_aux.conj().T)
    #v_aux_2 = 0.5 * (v_aux_2 + v_aux_2.conj().T)

    #print(f"v aux\n{np.real(v_aux)}")
    #print(f"v aux 2\n{np.real(v_aux_2)}")

    #print(f"h herm test: {np.allclose(h_full, h_full.conj().T)}")
    #print(f"v herm test: {np.allclose(v_full, np.conj(np.transpose(v_full, (2, 3, 0, 1))))}")

    #uses updated epsilon values from fock eigenvalues
    epsilon = fock_final_val

else:
    epsilon = np.diag(h_full)

def A_term(a_upper: int, a_lower: int, a_site: int)->np.array:
    """Generates the A commutator matrix based on upper and lower index and site given"""
    #are upper and lower indices needed if always shape A^a_p
    #if sites are all the same is site needed

    #horizontally combines the t_a_i amplitudes from a_site with identity matrix of size a
    A = np.hstack((-t_a_i_tensor[a_site], np.identity(a_upper)))
    return A

def B_term(b_upper: int, b_lower: int, b_site)->np.array:
    """Generates the B commutator matrix based on upper and lower index and site given"""
    #are indices needed (see A)
    #vertically combines identity matrix size i with t_a_i amplitudes from site_b
    B = np.vstack((np.identity(b_lower), t_a_i_tensor[b_site]))
    return B

def h_term(h_upper: int, h_lower: int, h_site: int)->np.array:
    """Slices h_full based on given indices"""
    #site seems unneeded due to all sites being the same
    a_h_shift = [i if a_check == a else 0 for a_check in (h_upper, h_lower)]
    return h_full[a_h_shift[0]:h_upper + a_h_shift[0], a_h_shift[1]:h_lower + a_h_shift[1]]

def v_term(v_upper_1: int, v_upper_2: int, v_lower_1: int, v_lower_2: int, v_site_1: int, v_site_2: int)->np.array:
    """Slices v_full based on given indices and sites"""

    if abs(v_site_1 - v_site_2) == 1 or abs(v_site_1 - v_site_2) == (sites - 1):
        a_v_shift = [i if a_check == a else 0 for a_check in (v_upper_1, v_upper_2, v_lower_1, v_lower_2)]
        return v_full[a_v_shift[0]:v_upper_1 + a_v_shift[0], a_v_shift[1]:v_upper_2 + a_v_shift[1], a_v_shift[2]:v_lower_1 + a_v_shift[2], a_v_shift[3]:v_lower_2 + a_v_shift[3]]
    else:
        return np.zeros((v_upper_1, v_upper_2, v_lower_1, v_lower_2))

def t_term(t_upper_1: int, t_upper_2: int, t_lower_1: int, t_lower_2: int, t_site_1: int, t_site_2: int)->np.array:
    """Slices t_ab_ij to give t amplitudes for 2 sites"""

    return t_ab_ij_tensor[t_site_1, t_site_2]

#"# noinspection SpellCheckingInspection" are for pycharm to ignore the einsum strings it thinks are typos  i.e. "plcd"
#not needed, I found the typo errors annoying

def residual_single(x_s:int)->np.array:
    """Calculates R^{a}_{i}(x) singles equation"""
    if HF and start_point == "sin":
        R_single = np.zeros((a, i), dtype = complex)
    else:
        R_single = np.zeros((a, i))

    R_single += np.einsum("ap, pq, qi->ai", A_term(a, p, x_s), h_term(p, q, x_s), B_term(q, i, x_s))

    for z_s in range (sites):
        if z_s != x_s:
            if i_method >= 1:
                # noinspection SpellCheckingInspection
                R_single += np.einsum("ap, plcd, cdil->ai", A_term(a, p, x_s), v_term(p, l, c, d, x_s, z_s), t_term(c, d, i, l, x_s, z_s))

            # noinspection SpellCheckingInspection
            R_single += np.einsum("ap, plqs, qi, sl->ai", A_term(a, p, x_s), v_term(p, l ,q, s, x_s, z_s), B_term(q, i, x_s), B_term(s, l, z_s))
    #print(f"single\n{R_single}")
    return R_single

def residual_double_sym(x_ds:int, y_ds:int)->np.array:
    """Calculates Rs^{ab}_{ij}(x < y) symmetric doubles equation"""
    if HF and start_point == "sin":
        R_double_symmetric = np.zeros((a, b, i, j), dtype = complex)
    else:
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
    if HF and start_point == "sin":
        R_double_non_symmetric_1 = np.zeros((a, b, i, j), dtype = complex)
    else:
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
    if HF and start_point == "sin":
        R_double_non_symmetric_2 = np.zeros((a, b, i, j), dtype=complex)
    else:
        R_double_non_symmetric_2 = np.zeros((a, b, i, j))

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
    """Calculates Rt^{ab}_{ij}(x < y) = Rs^{ab}_{ij}(x < y) + Rn^{ab}_{ij}(x < y) + Rn^{ba}_{ji}(y > x)total doubles equation"""

    return residual_double_sym(x_d, y_d) + residual_double_non_sym_1(x_d, y_d) + residual_double_non_sym_2(x_d, y_d)

def update_one(r_1_value:np.array)->np.array:
    """Calculates the iterative update value for t_1 amplitudes"""
    # the negative from the equation is in the while loop, approx line 425
    update_1 = np.zeros((a, i))

    for u_a_1 in range(a):
        for u_i_1 in range(i):
            update_1[u_a_1, u_i_1] = 1 / (epsilon[u_a_1 + i] - epsilon[u_i_1])

    return np.multiply(update_1, r_1_value)

def update_two(r_2_value:np.array)->np.array:
    """Calculates the iterative update value for t_2 amplitudes"""
    #the negative from the equation is in the while loop, approx line 429
    update_2 = np.zeros((a, b, i, j))

    for u_a_2 in range(a):
        for u_b_2 in range(b):
            for u_i_2 in range(i):
                for u_j_2 in range(j):
                    update_2[u_a_2, u_b_2, u_i_2, u_j_2] = 1 / (epsilon[u_a_2 + i] + epsilon[u_b_2 + i] - epsilon[u_i_2] - epsilon[u_j_2])

    return np.multiply(update_2, r_2_value)

def t_1_amplitude_csv():
    """Writes the t_1tensor to a csv file"""
    for t_1_site in range(sites):
        for t_1_a in range(a):
            for t_1_i in range(i):
                t_1_amplitudes_file.write(
                    f"{iteration}, {t_1_site}, {t_1_a + i}, {t_1_i}, {t_a_i_tensor[t_1_site, t_1_a, t_1_i]}\n")
    return None

def t_2_amplitude_csv():
    """Writes the t_2 tensor to a csv file"""
    for t_2_site_1 in range(sites):
        for t_2_site_2 in range(sites):
            for t_2_a in range(a):
                for t_2_b in range(b):
                    for t_2_i in range(i):
                        for t_2_j in range(j):
                            t_2_amplitudes_file.write(
                                f"{iteration}, {t_2_site_1}, {t_2_site_2}, {t_2_a + i}, {t_2_b + j}, {t_2_i}, {t_2_j}, {t_ab_ij_tensor[t_2_site_1, t_2_site_2, t_2_a, t_2_b, t_2_i, t_2_j]}\n")
    return None

if __name__ == "__main__":

    file_path_energy = "energy.csv"
    file_path_t_1_amplitudes = "t_1 amplitudes.csv"
    file_path_t_2_amplitudes = "t_2 amplitudes.csv"

    with open(file_path_energy, "w", encoding = "utf-8") as energy_file, \
         open(file_path_t_1_amplitudes, "w") as t_1_amplitudes_file, \
         open(file_path_t_2_amplitudes, "w") as t_2_amplitudes_file:

        energy_file.write("Iteration, Energy, ΔEnergy\n")
        t_1_amplitudes_file.write("iteration, site, a, i, value\n")
        t_2_amplitudes_file.write("iteration, site 1, site 2, a, b, i, j, value\n")

        iteration = 0
        if HF and start_point == "sin":
            single = np.zeros((sites, a, i), dtype = complex)
            double = np.zeros((sites, sites, a, b, i ,j), dtype = complex)
        else:
            single = np.zeros((sites, a, i))
            double = np.zeros((sites, sites, a, b, i, j))

        previous_energy = 0

        while True:

            energy = 0

            single[0] = residual_single(0)
            for y_site in range(1, sites):
                single[y_site] = single[0]
            #     #since 0 center site 1 + non-zero res, and sites - non-zero res
            #     #for 49 sites and 10 non-zero res
            #     #1 + 10 = 11
            #     #49 - 10 = 39
            #     #if y_site < 11 or y_site >= 39:
            #
                double[0, y_site] = residual_double_total(0, y_site)
                for x_site in range(1, sites):
                    double[x_site, (x_site + y_site) % sites] = double[0, y_site]

            # if HF:
            #     #middle = np.zeros((p, q))
            #
            #     #print(f"b term\n{B_term(s, l, 0)}")
            #     x_test = 2
            #     middle = h_term(p, q, x_test)
            #     h_val, h_vec = np.linalg.eigh(middle)
            #     for z_test in range(sites):
            #         if z_test != x_test:
            #             middle += np.einsum("plqs, sl ->pq", v_term(p, l, q, s, x_test, z_test), B_term(s, l, z_test))
            #     test = np.einsum("ap, pq, qi ->ai", A_term(a, p, x_test), middle, B_term(q, i, x_test))
            #     #print(f"middle\n {np.real(middle)}")
            #     middle_2 = h_term(p, q, x_test) + 2 * np.einsum("plqs, sl ->pq", v_term(p, l, q, s, x_test, 1), d_ij)
                #print(f"middle 2\n {np.real(middle_2)}")
                #print(f"v aux - middle\n{np.real(v_aux_2 - middle)}")
                #if iteration <= 1:
                    #print(f"fock test: {np.allclose(middle, fock_final)}")
                    #print(f"{fock_final - middle}")
                #print(f"single test: {np.allclose(test, single[x_test])}")
                #print(f"single test\n{np.real(test - single[x_test])}")

            one_max = single.flat[np.argmax(np.abs(single))]
            two_max = double.flat[np.argmax(np.abs(double))]

            print(f"1 max: {one_max}")
            print(f"2 max: {two_max}")

            #print(f"t1 max {np.max(abs(t_a_i_tensor))}")
            #print(f"t2 max {np.max(abs(t_ab_ij_tensor))}")

            if np.all(abs(single) <= threshold) and np.all(abs(double) <= threshold):
                break

            #CHANGE BACK TO 10
            if abs(one_max) >= 100 or abs(two_max) >= 100:
                raise ValueError("Diverges")

            # writes t1 amplitudes to csv file
            t_1_amplitude_csv()

            # writes t2 amplitudes to csv file
            t_2_amplitude_csv()

            t_a_i_tensor[0] -= update_one(single[0])
            for site_1 in range(1, sites):
                t_a_i_tensor[site_1] = t_a_i_tensor[0]
                t_ab_ij_tensor[0, site_1] -= update_two(double[0, site_1])
                for site_2 in range(1, sites):
                    t_ab_ij_tensor[site_2, (site_1 + site_2) % sites] = t_ab_ij_tensor[0, site_1]

            #energy calculations
            for site_x in range(sites):
                energy += np.einsum("ip, pi->", h_term(i, p, site_x), B_term(p, i, site_x)) * 0.5
                for site_y in range(site_x + 1, site_x + sites):
                    # noinspection SpellCheckingInspection
                    energy += np.einsum("ijab, abij->", v_term(i, j, a, b, site_x, site_y % sites), t_term(a, b, i, j, site_x, site_y % sites)) * 0.5
                    # noinspection SpellCheckingInspection
                    energy += np.einsum("ijpq, pi, qj->", v_term(i, j, p, q, site_x, site_y % sites), B_term(p, i, site_x), B_term(q, j, site_y % sites)) * 0.5

            delta_energy = energy - previous_energy
            previous_energy = energy

            iteration += 1
            print(f"Iteration #: {iteration}")
            print(f"Energy: {np.real(energy)}\n")

            energy_file.write(f"{iteration}, {energy}, {delta_energy}\n")

# if transform:
#     print(f"E_hf: {e_hf}")


# t_1_max = np.max(np.real(t_a_i_tensor[0]))
# t_2_max = np.max(np.real(t_ab_ij_tensor[0]))
# print(f"t_1_max = {t_1_max}")
# print(f"t_2_max = {t_2_max}")

# max_per_y = []
# for thing in range(sites):
#     max_per_y.append(float(np.max(np.real(t_ab_ij_tensor[int(sites/2), thing]))))
# print(f"max_per_y = {max_per_y}")
