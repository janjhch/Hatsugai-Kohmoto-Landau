import numpy as np
from scipy.optimize import root_scalar
from .HK_1D import rho_1d
from .HKLseparable_1D import GLS_1d, solve_GLS_1d_for_rho

t = 1
d = 1


# ===================
# UNMODIFIED HK-MODEL
# ===================



# For a given rho, find the correspondng mu
def find_mu_of_rho_hk(rho: float, U: float):
    bracket=(-2 * t, 2 * t + U)

    func_mu = lambda mu: rho_1d(mu, U) - rho
    result = root_scalar(func_mu, bracket=bracket, method='brentq')

    if result.converged:
        return result.root
    else:
        raise RuntimeError(f"Keine Nullstelle gefunden für rho={rho}")


# For a given rho, find the U_c which fulfills the conditions given in the thesis
# Uses Bisection as algorithm can otherwise diverge
def find_U_c_of_rho_hk(rho: float):
    if rho <= 1:
        func_U = lambda U: find_mu_of_rho_hk(rho, U) + 2 * t - U
        result = root_scalar(func_U, bracket=(0, 4), method='bisect')
    elif rho > 1:
        func_U = lambda U: find_mu_of_rho_hk(rho, U) - 2 * t
        result = root_scalar(func_U, bracket=(0, 4), method='bisect')
    
    if result.converged:
        return result.root
    else:
        print(result.flag)
        raise RuntimeError(f"Keine Nullstelle gefunden für rho={rho}")
    

# Given an array of rho values, calculate the corresponding critical interactions U_c,
# which are returned in the form of an array
def create_U_c_array_hk(rho_array: np.ndarray):
    U_c_list = []
    N = len(rho_array)
    i = 0

    for rho_val in rho_array:
        try:
            U_c_val = find_U_c_of_rho_hk(rho_val)
            U_c_list.append(U_c_val)
        except RuntimeError:
            U_c_list.append(np.nan)  

    U_c_array = np.array(U_c_list)

    return U_c_array


# Returns N samples of rho, U_c(rho) pairs
def phase_diagram_hk(N: int):
    mu_array = np.linspace(-2 * t * d, 2 * t * d, N)

    U_c_array = mu_array + 2 * t * d
    U_c_reversed = U_c_array[::-1]

    rho_list =[]
    for i in range(N):
        rho_val = rho_1d(mu_array[i], U_c_array[i])
        rho_list.append(rho_val)

    for i in range(N):
        rho_list.append(rho_list[i] + 1)

    U_c_array = np.concatenate([U_c_array, U_c_reversed])

    rho_array = np.array(rho_list)

    return rho_array, U_c_array


# ================================
# HK-MODEL WITH LANDAU INTERACTION
# ================================



def phase_diagram_landau(N: int, f_1: float):
    mu_arr = np.linspace(-2 * t * d, 2 * t * d, N)

    Uc_list = []
    rho_list = []

    i: int = 0

    for mu_val in mu_arr:
        print(f'\rProgress: {(i/N * 100):.1f}%{' ' * 20}', end="", flush=True)
        func_u = lambda U: mu_val + 2 * t * d * (1 + f_1 * solve_GLS_1d_for_rho(mu_val, U, 0, f_1)[1]) - U
        try:
            Uc_val = root_scalar(func_u, method='brentq', bracket=(0, 4 * t * d))
            rho_val = solve_GLS_1d_for_rho(mu_val, Uc_val.root, 0, f_1)[0]

            Uc_list.append(Uc_val.root)
            rho_list.append(rho_val)
        except ValueError:
            Uc_list.append(np.nan)
            rho_list.append(np.nan)
            print(f'Warning: Root finding failed at {i}-th point!')
        i += 1

    Uc_list = Uc_list[1:-1]
    rho_list = rho_list[1:-1]

    for i in range(N-2):
        rho_list.append(rho_list[i] + 1)

    Uc_arr = np.array(Uc_list)
    U_c_reversed = Uc_arr[::-1]
    Uc_arr = np.concatenate([Uc_arr, U_c_reversed])

    rho_arr = np.array(rho_list)

    Uc_arr = np.ma.masked_invalid(Uc_arr)
    rho_arr = np.ma.masked_invalid(rho_arr)

    return Uc_arr, rho_arr
