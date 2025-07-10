import numpy as np
import matplotlib.pyplot as plt

k_B=1.380649e-23

def beta_func(t: float)-> float:
    return 1/(k_B*t)

def Z(eig_val: np.ndarray, beta_val: float)->complex:
    exp_vector = np.exp(-eig_val * beta_val)
    return np.einsum('n->', exp_vector)


def P_n(eig_val: np.ndarray, beta_val: float)-> np.ndarray:
    return (1/Z(eig_val, beta_val)) * np.exp(-eig_val*beta_val)


def U(eig_val: np.ndarray, beta_val: float)-> complex:
    P_n_val = P_n(eig_val, beta_val)
    return (np.einsum('n->', P_n_val * eig_val))


def S(eig_val: np.ndarray, beta_val: float)-> complex:
    P_n_val = P_n(eig_val, beta_val)
    val = P_n_val * np.log(P_n_val)
    return  np.einsum('n->', -val)


def A(eig_val: np.ndarray, beta_val: float)-> complex:
    return (-1/beta_val) * np.log(Z(eig_val, beta_val))

def heat_capacity(eig_val: np.ndarray, beta_val: float):

    P_n_val = P_n(eig_val, beta_val)
    U_val = U(eig_val, beta_val)
    variance = np.sum(P_n_val * (eig_val - U_val)**2)

    C = k_B * beta_val**2 * variance
    return C

def generate_graphs(eig_val: np.ndarray, site: int):

    index = np.argsort(eig_val)
    eig_val = eig_val[index]
    beta_array = np.linspace(0, 5, 300)


    y_data_Z = np.array([np.log(Z(eig_val, x))/site for x in beta_array])

    plt.figure(figsize=(12, 6))
    plt.plot(beta_array, y_data_Z, label=f"{site} Sites")
    plt.xticks(beta_array[::12], rotation=45) 
    plt.xlabel("Beta-values")
    plt.ylabel("ln(Z)")
    plt.title("ln(Z) Evolution With Beta.")
    plt.legend()
    plt.show()

    beta_array = np.linspace(0, 3, 1000)
    y_data_U = np.array([U(eig_val, x)/site for x in beta_array])

    plt.figure(figsize=(12, 6))
    plt.plot(beta_array, y_data_U, label=f"{site} Sites")
    plt.xticks(beta_array[::40], rotation=45) 
    plt.xlabel("beta-values")
    plt.ylabel("Internal Thermal Energy")
    plt.title("Internal Thermal Energy Evolution With B.")
    plt.legend()
    plt.show()

    y_data_A = np.array([A(eig_val, x)/site for x in beta_array])

    plt.figure(figsize=(12, 6))
    plt.plot(beta_array, y_data_A, label=f"{site} Sites")
    plt.xticks(beta_array[::40], rotation=45) 
    plt.xlabel("B-values")
    plt.ylabel("Internal Thermal Energy")
    plt.title("Internal Thermal Energy Evolution With B.")
    plt.legend()
    plt.show()

    y_data_S = np.array([S(eig_val, x)/site for x in beta_array])

    plt.figure(figsize=(12, 6))
    plt.plot(beta_array, y_data_S, label=f"{site} Sites")
    plt.xticks(beta_array[::40], rotation=45) 
    plt.xlabel("B-values")
    plt.ylabel("Internal Thermal Energy")
    plt.title("Internal Thermal Energy Evolution With B.")
    plt.legend()
    plt.show()

    beta_array = np.linspace(0, 20, 1000)

    y_data_heat_capacity = np.array([heat_capacity(eig_val, x)/site for x in beta_array])

    plt.figure(figsize=(12, 6))
    plt.plot(beta_array, y_data_heat_capacity, label=f"{site} Sites")
    plt.xticks(beta_array[::40], rotation=45) 
    plt.xlabel("B-values")
    plt.ylabel("Heat Capacity")
    plt.title("Heat Capacity Evolution With B.")
    plt.legend()
    plt.show()