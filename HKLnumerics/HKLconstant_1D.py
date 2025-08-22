import numpy as np
from scipy.optimize import root_scalar
from .HK_1D import rho_1d, kappa_1d

t = 1
d = 1

def create_mu_array(N: int, U: float, f_0: float):
    return np.linspace(-2 * t, 2 * t + U + 2 * f_0, N)




# =================
# DENSITY
# =================

def rho_landau(rho: float, mu: float, U: float, f_0: float):
    return rho - rho_1d(mu - f_0 * rho, U)

def find_rho_of_mu_landau(mu: float, U: float, f_0: float, bracket=(0, 2)):
    ziel = lambda rho: rho_landau(rho, mu, U, f_0)
    result = root_scalar(ziel, method='brentq', bracket=bracket)

    if result.converged:
        return result.root
    else:
        raise RuntimeError(f"Keine Nullstelle gefunden für mu={mu}")
    
def create_rho_array(mu_array: np.ndarray, U: float, f_0: float):
    rho_list = []
    N = len(mu_array)
    i = 0

    for mu_val in mu_array:
        print(f'\rProgress: {(i / N * 100):.1f}%{' ' * 20}', end="", flush=True)
        i += 1
        try:
            rho = find_rho_of_mu_landau(mu_val, U, f_0)
            rho_list.append(rho)
        except RuntimeError:
            rho_list.append(np.nan)

    rho_array = np.array(rho_list)

    return rho_array






# ===============
# COMPRESSIBILITY
# ===============

def kappa_landau(rho: float, mu: float, U: float, f_0: float):
    return kappa_1d(mu - f_0 * rho, t, U) / (1 + f_0 * kappa_1d(mu - f_0 * rho,t ,U))

def create_kappa_array(mu_array: np.ndarray, rho_array: np.ndarray, U: float, f_0: float):
    kappa_list = []
    N = len(mu_array)

    for i in range(len(mu_array)):
        try:
            kappa_mu = kappa_landau(rho_array[i], mu_array[i], U, f_0)
            kappa_list.append(kappa_mu)
        except RuntimeError:
            kappa_list.append(np.nan)  

    kappa_array = np.array(kappa_list)

    return kappa_array




# ======
# ENERGY
# ======

# work in progress!

