import numpy as np
from scipy.optimize import root_scalar
from .HK_2D import rho_2d
from .HKLseparable_2D import solve_GLS_2d_for_rho

t = 1
d = 2

def create_rho_array(N):
    return np.linspace(0, 2, N)




# ===================
# UNMODIFIED HK-MODEL
# ===================




def find_mu_of_rho_hk(rho: float, U: float):
    bracket = (-2*t*d, U + 2*t*d) # minimal and maximal values of mu
    func_mu = lambda mu: rho_2d(mu, U) - rho
    result = root_scalar(func_mu, bracket=bracket, method='brentq')

    if result.converged:
        return result.root
    else:
        raise RuntimeError(f"Keine Nullstelle gefunden für rho={rho}")


def find_U_c_of_rho_hk(rho: float):
    if rho <= 1:
        func_U = lambda U: find_mu_of_rho_hk(rho, U) + 2*t*d - U
        result = root_scalar(func_U, bracket=(0, 4*t*d), method='bisect')
    elif rho > 1:
        func_U = lambda U: find_mu_of_rho_hk(rho, U) - 2*t*d
        result = root_scalar(func_U, bracket=(0, 4*t*d), method='bisect')

    if result.converged:
        return result.root
    else:
        print(result.flag)
        raise RuntimeError(f"Keine Nullstelle gefunden für rho={rho}")
    

def create_U_c_array_hk(rho_array: np.ndarray):
    U_c_list = []

    for rho_i in rho_array:
        print(f'\rProgress: {(rho_i/2 * 100):.1f}%{' ' * 20}', end="", flush=True)
        try:
            U_i = find_U_c_of_rho_hk(rho_i)
            U_c_list.append(U_i)
        except RuntimeError:
            U_c_list.append(np.nan)
            print(f"Keine Nullstelle gefunden für rho={rho_i}")

    U_c_array = np.array(U_c_list)

    return U_c_array

def phase_diagram_hk(N: int):
    mu_array = np.linspace(-2 * t * d, 2 * t * d, N)

    U_c_array = mu_array + 2 * t * d
    U_c_reversed = U_c_array[::-1]

    rho_list =[]
    for i in range(N):
        rho_val = rho_2d(mu_array[i], U_c_array[i])
        rho_list.append(rho_val)

    for i in range(N):
        rho_list.append(rho_list[i] + 1)

    U_c_array = np.concatenate([U_c_array, U_c_reversed])

    rho_array = np.array(rho_list)

    return rho_array, U_c_array



# =================================
# HK MODEL WITH LANDAU INTERACTIONS
# =================================


def phase_diagram_landau(N: int, f_1: float):
    mu_arr = np.linspace(-2 * t * d, 2 * t * d, N)

    Uc_list = []
    rho_list = []

    i: int = 0

    for mu_val in mu_arr:
        print(f'\rProgress: {(i/N * 100):.1f}%{' ' * 20}', end="", flush=True)
        func_u = lambda U: mu_val + 2 * t * d * (1 + f_1 * solve_GLS_2d_for_rho(mu_val, U, 0, f_1)[1]) - U
        Uc_val = root_scalar(func_u, method='brentq', bracket=(0, 4 * t * d))
        rho_val = solve_GLS_2d_for_rho(mu_val, Uc_val.root, 0, f_1)[0]

        Uc_list.append(Uc_val.root)
        rho_list.append(rho_val)
        i += 1

    print(f'\rProgress: {(i/N * 100):.1f}%{' ' * 20}', end="", flush=True)

    Uc_arr = np.array(Uc_list)
    rho_arr = np.array(rho_list)

    return Uc_arr, rho_arr