import numpy as np
from scipy import integrate

t = 1
d = 2

def create_mu_array(N: int, U: float):
    return np.linspace(-2 * t * d, 2 * t * d + U, N)

# =======
# DENSITY
# =======


def I_1(x: float):
    if np.abs(x) <= 2*t:
        result = np.heaviside(x + 2*t, 1) - (1 / np.pi) * np.arccos(x / (2*t))
    else:
        result = np.heaviside(x + 2*t, 1)
    return result

# I_2(x) = int_-pi^pi dk I_1(x + 2tcosk)
def I_2(x: float):
    def I_1_shifted(k):
        return I_1(x + 2*t*np.cos(k))
    
    integral_value = integrate.quad(I_1_shifted, -np.pi, np.pi)
    
    return integral_value[0] / (2*np.pi)

# rho = I_2(mu) + I_2(mu - U)
def rho_2d(mu: float, U: float):
    if U >= 0:
        return I_2(mu) + I_2(mu - U)
    elif U < 0:
        return 2 * I_2(mu - U/2)
    
def create_rho_array(mu_array: np.ndarray, U: float):
    rho_list = []
    N = len(mu_array)
    i = 0

    for mu_val in mu_array:
        print(f'\rProgress: {(i / N * 100):.1f}%{' ' * 20}', end="", flush=True)
        i += 1
        rho_val = rho_2d(mu_val, U)
        rho_list.append(rho_val)

    rho_array = np.array(rho_list)

    return rho_array




# ===============
# COMPRESSIBILITY
# ===============

def DOS_2D(E: float):
    def integrand_DOS_2d(k):
       if -1 <= E/(2*t) + np.cos(k) <= 1:
            integrand = 1 / np.sqrt(1 - (E/(2*t) + np.cos(k))**2)
            return integrand
       else:
            return 0
       
    int_value = integrate.quad(integrand_DOS_2d, 0, np.pi)
    full_DOS = int_value[0] / (4 * np.pi**2 * t)
    return full_DOS

def kappa_2d(mu: float, U: float):
    if U >= 0:
        return (DOS_2D(mu) + DOS_2D(mu - U))
    else:
        return 2 * DOS_2D(mu - U/2)
    
# Given an array of mu values, returns an array of corresponding kappa values  
def create_kappa_array(mu_array: np.ndarray, U: float):
    kappa_list = []
    N = len(mu_array)
    i = 0

    for mu_val in mu_array:
        print(f'\rProgress: {(i / N * 100):.1f}%{' ' * 20}', end="", flush=True)
        i += 1
        kappa_val = kappa_2d(mu_val, U)
        kappa_list.append(kappa_val)

    kappa_array = np.array(kappa_list)

    return kappa_array




# ==============
# ENERGY DENSITY
# ==============

