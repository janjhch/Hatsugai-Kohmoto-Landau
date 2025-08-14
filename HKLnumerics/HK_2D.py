# NOTE:
# This file gives functions to calculate thermodynamic quantities of an unmodified HK model in two dimensions.
# There are significantly less comments in this file. The naming of the functions should however be
# self-explanatory given the comments in the file 'HK_1D.py' and the function defined in the thesis.

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

def I_2(x: float):
    def I_1_shifted(k):
        return I_1(x + 2*t*np.cos(k))
    
    integral_value = integrate.quad(I_1_shifted, -np.pi, np.pi)
    
    return integral_value[0] / (2*np.pi)

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

def J_1(x: float):
    if -1 <= x/(2*t) <= 1:
        result = - np.sqrt((2*t)**2 - x**2) /np.pi
        return result
    else:
        return 0
    
def J_2(x: float):
    def J_1_shifted(k):
        return J_1(x + 2*t*np.cos(k))
    
    result_integral = integrate.quad(J_1_shifted, -np.pi, np.pi)

    return result_integral[0] / np.pi

def energy_2d(mu: float, U: float):
    if U >= 0:
        return J_2(mu) + J_2(mu - U) + U*I_2(mu - U)
    else:
        return 2*J_2(mu - U/2) + U*I_2(mu - U/2)
    
def create_energy_array(mu_array: np.ndarray, U: float):
    e_list = []
    N = len(mu_array)
    i = 0

    for mu_val in mu_array:
        print(f'\rProgress: {(i / N * 100):.1f}%{' ' * 20}', end="", flush=True)
        i += 1
        e_val = energy_2d(mu_val, U)
        e_list.append(e_val)

    e_array = np.array(e_list)

    return e_array

