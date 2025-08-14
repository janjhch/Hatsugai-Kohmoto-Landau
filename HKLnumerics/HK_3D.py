import numpy as np
from scipy import integrate

t = 1
d = 3

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

def I_3(x: float):
    I_1_shifted = lambda k1, k2: I_1(x + 2 * t * (np.cos(k1) + np.cos(k2)))

    integral_val = integrate.dblquad(I_1_shifted, -np.pi, np.pi, -np.pi, np.pi)

    return_val = integral_val[0] / ((2 * np.pi)**2)

    return return_val

def rho_3d(mu: float, U: float):
    if U >= 0:
        return I_3(mu) + I_3(mu - U)
    elif U < 0:
        return 2 * I_3(mu - U/2)

def create_rho_array(mu_array: np.ndarray, U: float):
    rho_list = []
    N = len(mu_array)
    i = 0

    for mu_val in mu_array:
        print(f'\rProgress: {(i / N * 100):.1f}%{' ' * 20}', end="", flush=True)
        i += 1
        rho_val = rho_3d(mu_val, U)
        rho_list.append(rho_val)

    rho_array = np.array(rho_list)

    return rho_array




# ===============
# COMPRESSIBILITY
# ===============

def DOS_1(E: float):
    if np.abs(E) < (2 * t):
        return 2 / (np.pi * np.sqrt((2 * t)**2 - E**2))
    else:
        return 0
    
def DOS_3(E: float):
    DOS_1_shifted = lambda k1, k2: DOS_1(E + (2 * t) * (np.cos(k1) + np.cos(k2)))

    integral_val = integrate.dblquad(DOS_1_shifted, -np.pi, np.pi, -np.pi, np.pi)

    return_val = integral_val[0] / ((2 * np.pi)**2)

    return return_val

def kappa_3d(mu: float, U: float):
    if U >= 0:
        return DOS_3(mu) + DOS_3(mu - U)
    else:
        return 2 * DOS_3(mu - U / 2)
    
def create_kappa_array(mu_array: np.ndarray, U: float):
    kappa_list = []
    N = len(mu_array)
    i = 0

    for mu_val in mu_array:
        print(f'\rProgress: {(i / N * 100):.1f}%{' ' * 20}', end="", flush=True)
        i += 1
        kappa_val = kappa_3d(mu_val, U)
        kappa_list.append(kappa_val)

    kappa_array = np.array(kappa_list)

    return kappa_array




# ==============
# ENERGY DENSITY
# ==============

def J_1(x: float):
    if np.abs(x) < (2 * t):
        return - np.sqrt((2 * t)**2 - x**2) / np.pi
    else:
        return 0
    
def J_3(x: float):
    J_1_shifted = lambda k1, k2: J_1(x + (2 * t) * (np.cos(k1) + np.cos(k2)))

    integral_val = integrate.dblquad(J_1_shifted, -np.pi, np.pi, -np.pi, np.pi)

    return_val = d * integral_val[0] / ((2 * np.pi)**2)

    return return_val

def energy_3d(mu: float, U: float):
    if U >= 0:
        return J_3(mu) + J_3(mu - U) + U * I_3(mu - U)
    else: 
        return 2 * J_3(mu - U / 2) + U * I_3(mu - U / 2)
    
def create_energy_array(mu_array: np.ndarray, U: float):
    e_list = []
    N = len(mu_array)
    i = 0

    for mu_val in mu_array:
        print(f'\rProgress: {(i / N * 100):.1f}%{' ' * 20}', end="", flush=True)
        i += 1
        e_val = energy_3d(mu_val, U)
        e_list.append(e_val)

    e_array = np.array(e_list)

    return e_array