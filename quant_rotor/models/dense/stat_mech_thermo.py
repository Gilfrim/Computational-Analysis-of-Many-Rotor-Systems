import matplotlib.pyplot as plt
import numpy as np

k_B = 1.380649e-23
"""float: Boltzmann constant in J·K⁻¹ (SI exact value).
See: https://en.wikipedia.org/wiki/Boltzmann_constant
"""


def beta_func(t: float) -> float:
    """
    Inverse temperature β = 1 / (k_B * T).

    Parameters
    ----------
    t : float
        Absolute temperature T in kelvin (K). Must be > 0.

    Returns
    -------
    float
        β = 1 / (k_B * T) in J⁻¹.

    Notes
    -----
    - In statistical mechanics, β is the Lagrange multiplier for energy in
      the canonical ensemble.
    - For k_B = 1 (natural units), β = 1 / T.

    References
    ----------
    .. Wikipedia::
       - Inverse temperature: https://en.wikipedia.org/wiki/Inverse_temperature
       - Boltzmann constant: https://en.wikipedia.org/wiki/Boltzmann_constant
    """
    if t <= 0:
        raise ValueError("Temperature must be positive.")
    return 1.0 / (k_B * t)


def Z(eig_val: np.ndarray, beta_val: float) -> float:
    """
    Partition function Z(β) for a set of energy eigenvalues.

    Parameters
    ----------
    eig_val : np.ndarray
        1D array of energy eigenvalues E_n (in joules).
    beta_val : float
        Inverse temperature β = 1 / (k_B * T) (in J⁻¹).

    Returns
    -------
    float
        Partition function Z = Σ_n exp(-β E_n). Real and non-negative.

    References
    ----------
    .. Wikipedia::
       - Partition function (statistical mechanics):
         https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)
       - Canonical ensemble:
         https://en.wikipedia.org/wiki/Canonical_ensemble
    """
    exp_vector = np.exp(-eig_val * beta_val)
    return float(np.einsum("n->", exp_vector))


def P(eig_val: np.ndarray, beta_val: float) -> np.ndarray:
    """
    Boltzmann probability distribution over eigenstates in the canonical ensemble.

    Parameters
    ----------
    eig_val : np.ndarray
        1D array of energy eigenvalues E_n (J).
    beta_val : float
        Inverse temperature β (J⁻¹).

    Returns
    -------
    np.ndarray
        1D array p_n = exp(-β E_n) / Z with Σ_n p_n = 1 (up to numerical error).

    Notes
    -----
    - For extreme βE values, numerical underflow/overflow can occur. If needed,
      stabilize with a log-sum-exp trick:
      shift = E_min; p_n ∝ exp(-β (E_n - E_min)).

    References
    ----------
    .. Wikipedia::
       - Boltzmann distribution: https://en.wikipedia.org/wiki/Boltzmann_distribution
       - Canonical ensemble: https://en.wikipedia.org/wiki/Canonical_ensemble
    """
    z = Z(eig_val, beta_val)
    if z == 0 or not np.isfinite(z):
        raise FloatingPointError("Partition function is zero or non-finite.")
    return np.exp(-eig_val * beta_val) / z


def U(eig_val: np.ndarray, beta_val: float) -> float:
    """
    Internal energy ⟨E⟩ in the canonical ensemble.

    Parameters
    ----------
    eig_val : np.ndarray
        1D array of energies E_n (J).
    beta_val : float
        Inverse temperature β (J⁻¹).

    Returns
    -------
    float
        Internal energy U = Σ_n p_n E_n (J).

    References
    ----------
    .. Wikipedia::
       - Internal energy (statistical mechanics):
         https://en.wikipedia.org/wiki/Internal_energy#Statistical_mechanics
       - Canonical ensemble:
         https://en.wikipedia.org/wiki/Canonical_ensemble
    """
    p = P(eig_val, beta_val)
    return float(np.einsum("n->", p * eig_val))


def S(eig_val: np.ndarray, beta_val: float) -> float:
    """
    Gibbs (Shannon) entropy S = -Σ_n p_n log p_n for the canonical ensemble.

    Parameters
    ----------
    eig_val : np.ndarray
        1D array of energies E_n (J).
    beta_val : float
        Inverse temperature β (J⁻¹).

    Returns
    -------
    float
        Entropy S in nats (natural logarithm). Multiply by k_B for physical units,
        or interpret this as S/k_B if your convention folds k_B into β.

    Notes
    -----
    - This uses the natural log. If you prefer bits, divide by ln(2).
    - We treat 0·log 0 := 0 by masking zeros to avoid NaNs.

    References
    ----------
    .. Wikipedia::
       - Entropy (statistical thermodynamics):
         https://en.wikipedia.org/wiki/Entropy_(statistical_thermodynamics)
       - Shannon entropy: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    p = P_n(eig_val, beta_val)
    # Avoid 0*log(0) = NaN by masking zeros (limit is 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(p > 0, p * np.log(p), 0.0)
    return float(-np.einsum("n->", term))


def A(eig_val: np.ndarray, beta_val: float) -> float:
    """
    Helmholtz free energy A = -(1/β) log Z for the canonical ensemble.

    Parameters
    ----------
    eig_val : np.ndarray
        1D array of energies E_n (J).
    beta_val : float
        Inverse temperature β (J⁻¹).

    Returns
    -------
    float
        Helmholtz free energy A (J).

    References
    ----------
    .. Wikipedia::
       - Free energy: https://en.wikipedia.org/wiki/Free_energy
       - Helmholtz free energy: https://en.wikipedia.org/wiki/Helmholtz_free_energy
       - Partition function: https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)
    """
    return float((-1.0 / beta_val) * np.log(Z(eig_val, beta_val)))


def heat_capacity(eig_val: np.ndarray, beta_val: float) -> float:
    """
    Heat capacity C via energy fluctuations in the canonical ensemble.

    Uses the fluctuation–dissipation relation:
        Var(E) = ⟨E²⟩ - ⟨E⟩²,   C = k_B * β² * Var(E)

    Parameters
    ----------
    eig_val : np.ndarray
        1D array of energies E_n (J).
    beta_val : float
        Inverse temperature β (J⁻¹).

    Returns
    -------
    float
        Heat capacity C (J/K).

    Notes
    -----
    - Equivalent expressions:
        C = d⟨E⟩/dT
        C = (Var(E)) / (k_B T²)  = k_B β² Var(E)
      since β = 1/(k_B T).
    - Numerically, this uses the variance under the Boltzmann distribution.

    References
    ----------
    .. Wikipedia::
       - Heat capacity: https://en.wikipedia.org/wiki/Heat_capacity
       - Fluctuation–dissipation theorem:
         https://en.wikipedia.org/wiki/Fluctuation%E2%80%93dissipation_theorem
       - Canonical ensemble:
         https://en.wikipedia.org/wiki/Canonical_ensemble
    """
    p = P(eig_val, beta_val)
    U_r = U(eig_val, beta_val)
    variance = float(np.sum(p * (eig_val - U_r) ** 2))
    return float(k_B * (beta_val ** 2) * variance)

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
