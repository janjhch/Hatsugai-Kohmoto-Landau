import numpy as np
from scipy.optimize import root_scalar, minimize
from .HK_1D import rho_1d
from .HKLseparable_1D import GLS_1d

t = 1
d = 1

def create_rho_array(N):
    return np.linspace(0, 2, N)


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





# ================================
# HK-MODEL WITH LANDAU INTERACTION
# ================================





def minimize_GLS_norm_1d_wrt_mu(rho: float, U: float, f_0: float, f_1: float, fit_param=1):
    a = fit_param
    guess_mu = (U / 2 + 2) * rho - 2
    guess_e = a * (rho**2 - 2 * rho)
    guess = [guess_mu, guess_e]

    # def x = [mu, e_tilde]
    GLS_reduced = lambda x: GLS_1d(rho, x[0], U, x[1], f_0, f_1)
    GLS_reduced_norm = lambda x: GLS_reduced(x)[0]**2 + GLS_reduced(x)[1]**2

    sol = minimize(GLS_reduced_norm, guess, method='L-BFGS-B')
    # Should return list [mu, e_tilde] for any given rho
    return sol.x


def find_U_c_of_rho_landau(rho: float, f_0: float, f_1: float):
    if rho <= 1:
        def func_U(U):
            mu, e = minimize_GLS_norm_1d_wrt_mu(rho, U, f_0, f_1)
            return mu + 2 * t * (1 + f_1 * e) - U
        result = root_scalar(func_U, x0=4*rho, method='newton', xtol=1e-3)
    elif rho > 1:
        def func_U(U):
            mu, e = minimize_GLS_norm_1d_wrt_mu(rho, U, f_0, f_1)
            return mu - 2 * t * (1 + f_1 * e)
        result = root_scalar(func_U, x0=(8-4*t*d*rho), method='newton', xtol=1e-3)

    if result.converged:
        return result.root
    else:
        print(result.flag)
        raise RuntimeError(f"Keine Nullstelle gefunden für rho={rho}")


def create_U_c_array_landau(rho_array: np.ndarray, f_0: float, f_1: float):
    U_c_list = []

    for rho_i in rho_array:
        print(f'\rProgress: {(rho_i/2 * 100):.1f}%{' ' * 20}', end="", flush=True)
        try:
            U_i = find_U_c_of_rho_landau(rho_i, f_0, f_1)
            U_c_list.append(U_i)
        except RuntimeError:
            U_c_list.append(np.nan)
            print(f"Keine Nullstelle gefunden für rho={rho_i}")

    U_c_array = np.array(U_c_list)

    return U_c_array

