# NOTE:
# This file gives functions to calculate thermodynamic quantities of an unmodified HK model in one dimension.
# All functions take as base values arrays of the chemical potential mu.
# An array of N mu values corresponding to rho =[0, 2] for any interaction U can be obtained via the function
#   create_mu_array(N, U)
# Importantly, it is not simply possible to create an array of rho values and calculate the rest of the quantities
# here. You must always take an array of mu values and find the rho values that are mapped to those mus values via 
# the function
#   create_rho_array(mu_array, U)


import numpy as np

t = 1


# ================================================================
# OBTAIN DENSITY NUMERICALLY VIA INTEGRATION OF OCCUPATION NUMBERS
# ================================================================

# kinetic energy eigenvalues
def eps_k(t, k):
    return -2 * t * np.cos(k)

# define the integrand of the integration for the density
def rho_integrand(U, t, k, mu):
    if U >= 0:
        return (np.heaviside(mu - eps_k(t, k), 1) + np.heaviside(mu - eps_k(t, k) - U, 1)) / (2 * np.pi)
    elif U < 0:
        return (2 * np.heaviside(mu - eps_k(t, k) - (U / 2), 1)) / (2 * np.pi)
    
# Given an array of mu values, returns a corresponding array of rho values using trapezoidal integration 
def find_values_for_rho_numerically(U: float, no_of_points: int, integration_points: int, mu_array: np.ndarray):
    # define the integration space [-pi, pi]
    k_array = np.linspace(-np.pi, np.pi, integration_points)

    # define the array in which the values for rho will be stored in, this will be the return value
    rho_array = np.empty(no_of_points)  
  
    # calculate the integral (4.7) for all values in mu_array
    for i in range(no_of_points):
        # calculate the integrand for the given mu
        integrand = np.empty(integration_points)
        for k_index in range(integration_points):
            integrand[k_index] = rho_integrand(U, t, k_array[k_index], mu_array[i])
        
        # calculate the integral to find the rho that corresponds to the chosen mu in mu_index
        rho_array[i] = np.trapezoid(integrand, k_array)
        
    return rho_array





# ====================================
# OBTAIN DENSITY VIA ANALYTIC SOLUTION
# ====================================

# Analytic expression provided in HK-Paper (WRONG!)
def rho_ana_hk(U, t, mu):
    return 2 - ((np.heaviside((mu - U)/(2*t), 1.0) - np.heaviside((mu - U)/(2*t) - 1, 1.0)) * np.arccos((mu - U)/(2*t)) 
                 + (np.heaviside(mu/(2*t), 1.0) - np.heaviside(mu/(2*t) - 1, 1.0)) * np.arccos(mu/(2*t)))/ np.pi

# My own analytic expression. I_1 is the general solution to the integral I_1 presented in the thesis.
# rho_1D can be expressed only using the solution of I_1 for different arguments
def I_1(x):
    if np.abs(x) <= 2*t:
        result = np.heaviside(x + 2*t, 1) - (1 / np.pi) * np.arccos(x / (2*t))
    else:
        result = np.heaviside(x + 2*t, 1)
    return result
    
def rho_1d(mu: float, U: float):
    if U >= 0:
        return I_1(mu) + I_1(mu - U)
    elif U < 0:
        return 2 * I_1(mu - U/2)

# Given an array of mu values, returns an array of corresponding rho values    
def create_rho_array(mu_array: np.ndarray, U: float):
    rho_list = []
    N = len(mu_array)
    i = 0

    for mu_val in mu_array:
        print(f'\rProgress: {(i / N * 100):.1f}%{' ' * 20}', end="", flush=True)
        i += 1
        rho_val = rho_1d(mu_val, U)
        rho_list.append(rho_val)

    rho_array = np.array(rho_list)

    return rho_array

# Simply creates array of N mu values for any given U
# Note that this DOES NOT give mu(rho) for an array of rho values!
def create_mu_array(N: int, U: float):
    return np.linspace(-2 * t, 2 * t + U, N)




# ===============
# COMPRESSIBILITY
# ===============

# Analytic solution for the density of states in one dimension
def DOS_1d(E: float):
    if np.abs(E) < 2 * t:
        return 1 / (np.pi * np.sqrt((2 * t)**2 - E**2))
    else:
        return 0
    
def kappa_1d(mu: float, U: float):
    if 0 <= U:
        return DOS_1d(mu) + DOS_1d(mu - U)
    else:
        return 2 * DOS_1d(mu - U / 2)
    
# Given an array of mu values, returns an array of corresponding kappa values  
def create_kappa_array(mu_array: np.ndarray, U: float):
    kappa_list = []
    N = len(mu_array)
    i = 0

    for mu_val in mu_array:
        print(f'\rProgress: {(i / N * 100):.1f}%{' ' * 20}', end="", flush=True)
        i += 1
        kappa_val = kappa_1d(mu_val, U)
        kappa_list.append(kappa_val)

    kappa_array = np.array(kappa_list)

    return kappa_array






# ========================================
# ENERGY DENSITY VIA NUMERICAL INTEGRATION
# ========================================

def energy_integrand(U, t, k, mu):
    if U >= 0:
        return (eps_k(t, k) * np.heaviside(mu - eps_k(t, k), 1) + ((eps_k(t, k) + U) * np.heaviside(mu - eps_k(t, k) - U, 1))) / (2 * np.pi)
    elif U < 0:
        return ((2 * eps_k(t, k) + U) * np.heaviside(mu - eps_k(t, k) - (U / 2), 1)) / (2 * np.pi)
    
# Given an array of mu values, find corresponding energies using trapezoidal integration of energy_integrand
def find_values_for_energy_numerically(U: float, no_of_points: int, integration_points: int, mu_array: np.ndarray):
    # define the integration space [-pi, pi]
    k_array = np.linspace(-np.pi, np.pi, integration_points)

    # define the array in which the values for rho will be stored in, this will be the return value
    energy_array = np.empty(no_of_points)  
  
    # calculate the integral (4.7) for all values in mu_array
    for i in range(no_of_points):
        # calculate the integrand for the given mu
        integrand = np.empty(integration_points)
        for k_index in range(integration_points):
            integrand[k_index] = energy_integrand(U, t, k_array[k_index], mu_array[i])
        
        # calculate the integral to find the rho that corresponds to the chosen mu in mu_index
        energy_array[i] = np.trapezoid(integrand, k_array)
        
    return energy_array, integrand





# ====================================
# ENERGY DENSITY VIA ANALYTIC SOLUTION
# ====================================

# Note that this is not optimized to handle numpy arrays
def J_1(x: float):
    if -1 <= x/(2*t) <= 1:
        result = - np.sqrt((2*t)**2 - x**2) /np.pi
        return result
    else:
        return 0
      
def energy_1d(mu: float, U: float):
    if U >= 0:
        return J_1(mu) + J_1(mu - U) + U*I_1(mu - U)
    else:
        return 2*J_1(mu - U/2) + U*I_1(mu - U/2)
    
# Same story as before: Returns array of energy values corresponding to given mu values
def create_energy_array(mu_array: np.ndarray, U: float):
    e_list = []
    N = len(mu_array)
    i = 0

    for mu_val in mu_array:
        print(f'\rProgress: {(i / N * 100):.1f}%{' ' * 20}', end="", flush=True)
        i += 1
        e_val = energy_1d(mu_val, U)
        e_list.append(e_val)

    e_array = np.array(e_list)

    return e_array


