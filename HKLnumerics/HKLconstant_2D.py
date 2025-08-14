import numpy as np
from scipy import integrate
from scipy.optimize import root_scalar
from .HK_2D import rho_2d, kappa_2d

t = 1
d = 2

def create_mu_array(N: int, U: float, f_0: float):
    return np.linspace(-2 * t, 2 * t + U + 2 * f_0, N)



# =======
# DENSITY
# =======

def find_rho_of_mu_landau(mu: float, U: float, f_0: float):
    bracket = (0, 2) # minimal and maximal values of rho
    func_rho = lambda rho: rho_2d(mu - f_0 * rho, U) - rho
    result = root_scalar(func_rho, bracket=bracket, method='brentq')

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
            rho = find_rho_of_mu_landau(mu_val, t, U, f_0)
            rho_list.append(rho)
        except RuntimeError:
            rho_list.append(np.nan)

    rho_array = np.array(rho_list)

    return rho_array




# ===============
# COMPRESSIBILITY
# ===============

def kappa_landau_2d(rho: float, mu: float, U: float, f_0: float):
    kappa_hk_renorm = kappa_2d(mu - f_0 * rho, U)
    return kappa_hk_renorm / (1 + f_0 * kappa_hk_renorm)

def create_kappa_array(mu_array: np.ndarray, rho_array: np.ndarray, U: float, f_0: float):
    kappa_list = []
    N = len(mu_array)

    for i in range(len(mu_array)):
        try:
            kappa_mu = kappa_landau_2d(rho_array[i], mu_array[i], U, f_0)
            kappa_list.append(kappa_mu)
        except RuntimeError:
            kappa_list.append(np.nan)  

    kappa_array = np.array(kappa_list)

    return kappa_array

# Here are functions to invert the process, i.e. find, for a given kappa, the corresponding rho (or mu).
# This was used to examine the effects of a constant Landau interaction on the van-Hove singularities in
# 2d, which diverge logarithmically. You must first calculate kappa_HK and then calculate from that the
# renormalized compressibility (If you do this, check this step whether it is logical or not, I am no
# longer sure).

def find_mu_of_kappa_hk(kappa: float, U: float):
    bracket = (-t*d, 0)
    func_kappa = lambda mu: kappa_2d(mu, U) - kappa
    result = root_scalar(func_kappa, bracket=bracket, method='brentq')

    if result.converged:
        return result.root
    else:
        raise RuntimeError(f"Keine Nullstelle gefunden für kappa={kappa}")
    
def create_rho_list_of_kappa_values(kappa_array: np.ndarray, U: float):
    rho_list = []
    kappa_max = max(kappa_array)

    for kappa_i in kappa_array:
        print(f'\rProgress: {(kappa_i/kappa_max * 100):.1f}%{' ' * 20}', end="", flush=True)
        try:
            mu_i = find_mu_of_kappa_hk(kappa_i, U)
            rho_i = rho_2d(mu_i, U)
            rho_list.append(rho_i)
        except RuntimeError:
            rho_list.append(np.nan)
            print(f"Keine Nullstelle gefunden für kappa={kappa_i}")

    return rho_list

def create_kappa_renorm_list(kappa_array: np.ndarray, f_0: float):
    kappa_renorm_list = []
    for kappa_i in kappa_array:
        # Do not know if this step is correct, since there is a shift in the variables?
        kappa_renorm = kappa_i / (1 + f_0 * kappa_i)
        kappa_renorm_list.append(kappa_renorm)

    return kappa_renorm_list




# ======
# ENERGY
# ======

# work in progress!
