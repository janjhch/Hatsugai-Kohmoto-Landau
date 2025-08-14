# IMPORTANT:
# There are two ways the integrals are implemented here, once using numpy's trapezoidal algorithm
# and once using scipy's integrate functions. It it HIGHLY RECCOMENDED to only use the functions 
# not containing the scipy integration, as it is immensely inefficient. The functions create_...
# all use the numpy integration anyway. The scipy function are only here for completeness, in 
# principle the accurcy of trapezoidal integration can be adjusted via N
# error(trapezoidal) = (4pi / N)^3 (roughly)

import numpy as np
from scipy import integrate
from scipy.optimize import root_scalar

t = 1
d = 3

N = 1000

# np.where(if statement, if true, if false)
def I_1(x: np.ndarray):
    return np.where(np.abs(x) < (2 * t),
             np.heaviside(x + 2*t, 1) - (1 / np.pi) * np.arccos(np.clip(x / (2*t), -1, 1)),
             np.heaviside(x + 2*t, 1))

k1_array = np.linspace(-np.pi, np.pi, N)
k2_array = np.linspace(-np.pi, np.pi, N)
k1k1, k2k2 = np.meshgrid(k1_array, k2_array)

def I_3(x):
    shifted_I_1 = x + 2 * t * (np.cos(k1k1) + np.cos(k2k2))
    func_val_grid = I_1(shifted_I_1)

    intermediate = np.trapezoid(func_val_grid, k2_array, axis=1)
    result = np.trapezoid(intermediate, k1_array)

    return result / ((2 * np.pi)**2)


# DO NOT USE!
def I_1_scipy(x):
    if np.abs(x) < 2*t:
        result = np.heaviside(x + 2*t, 1) - (1 / np.pi) * np.arccos(x / (2*t))
    else:
        result = np.heaviside(x + 2*t, 1)
    return result
# DO NOT USE!
def I_3_scipy(x):
    I_1_shifted = lambda k1, k2: I_1_scipy(x + 2 * t * (np.cos(k1) + np.cos(k2)))

    integral_val = integrate.dblquad(I_1_shifted, -np.pi, np.pi, -np.pi, np.pi, epsabs=1e-4)

    return_val = integral_val[0] / ((2 * np.pi)**2)

    return return_val


def rho_3d(mu: float, U: float):
    if U >= 0:
        return I_3(mu) + I_3(mu - U)
    elif U < 0:
        return 2 * I_3(mu - U/2)

def find_mu_of_rho(rho: float, U: float):
    bracket = (-2*t*d, U + 2*t*d) # minimal and maximal values of mu
    func_mu = lambda mu: rho_3d(mu, U) - rho
    result = root_scalar(func_mu, bracket=bracket, method='brentq', xtol=1e-4)

    if result.converged:
        return result.root
    else:
        raise RuntimeError(f"Keine Nullstelle gefunden für rho={rho}")
    
def find_U_c_of_rho(rho: float):
    if rho <= 1:
        func_U = lambda U: find_mu_of_rho(rho, U) + 2*t*d - U
        result = root_scalar(func_U, bracket=(0, 4*t*d), method='brentq', xtol=1e-3)
    elif rho > 1:
        func_U = lambda U: find_mu_of_rho(rho, U) - 2*t*d
        result = root_scalar(func_U, bracket=(0, 4*t*d), method='brentq', xtol=1e-3)

    if result.converged:
        return result.root
    else:
        print(result.flag)
        raise RuntimeError(f"Keine Nullstelle gefunden für rho={rho}")

# Given an array of rho values, returns array of corresponding critical interactions for HK Model    
def create_U_c_array_hk(rho_array: np.ndarray):
    U_c_list = []

    for rho_i in rho_array:
        print(f'\rProgress: {(rho_i/2 * 100):.1f}%{' ' * 20}', end="", flush=True)
        try:
            U_i = find_U_c_of_rho(rho_i)
            U_c_list.append(U_i)
        except RuntimeError:
            U_c_list.append(np.nan)
            print(f"Keine Nullstelle gefunden für rho={rho_i}")

    U_c_array = np.array(U_c_list)

    return U_c_array


# HK MODEL WITH LANDAU INTERACTIONS?