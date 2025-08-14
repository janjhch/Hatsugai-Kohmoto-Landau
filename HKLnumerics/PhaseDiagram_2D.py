import numpy as np
from scipy.optimize import root_scalar
from .HK_2D import rho_2d

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




# HK MODEL WITH LANDAU INTERACTIONS?