import numpy as np
from scipy import integrate
from scipy.optimize import root

t = 1
d = 2

def create_mu_array(N: int, U: float, f_0: float):
    return np.linspace(-2 * t * d, 2 * t * d + U + 2 * f_0, N)

# I_1 with two variables
def I_1(x: float, y: float):
    if np.abs(x / (2 * t * y)) < 1:
        return np.heaviside(x + 2 * t * y, 1) - np.arccos(x / (2 * t * y)) / np.pi
    else:
        return np.heaviside(x + 2 * t * y, 1)
    
# J_1 with two varibales
def J_1(x: float, y: float):
    if np.abs(x / (2 * t * y)) < 1:
        return - np.sqrt((2 * t)**2 - (x / y)**2) / np.pi
    else:
        return 0

def I_2(x: float, y: float):
    I_1_shifted = lambda k: I_1(x + 2 * t * y * np.cos(k), y)

    int_val = integrate.quad(I_1_shifted, -np.pi, np.pi)

    return int_val[0] / (2 * np.pi)

def J_2(x: float, y: float):
    J_1_shifted = lambda k: J_1(x + 2 * t * y * np.cos(k), y)

    int_val = integrate.quad(J_1_shifted, -np.pi, np.pi)

    return d * int_val[0] / (2 * np.pi)

def GLS_2d(rho: float, mu: float, U: float, e_tilde: float, f_0: float, f_1: float):
    eq1 = rho - (I_2(mu - f_0 * rho, 1 + f_1 * e_tilde) + I_2(mu - U - f_0 * rho, 1 + f_1 * e_tilde))
    eq2 = e_tilde - (J_2(mu - f_0 * rho, 1 + f_1 * e_tilde) + J_2(mu - U - f_0 * rho, 1 + f_1 * e_tilde))
    return [eq1, eq2]

# Do not solve SOE for mu, but for rho
def solve_GLS_2d_for_rho(mu: float, U: float, f_0: float, f_1: float):
    # def x = [rho, e_tilde]
    GLS_reduced = lambda x: GLS_2d(x[0], mu, U, x[1], f_0, f_1)

    # Guess something near a possible solution
    if U <= 4 * t * d:
        # Model as one straight line through start and endpoint
        guess_rho = (2 / (4 * t * d + U)) * mu + 4 * t / (4 * t * d + U)
    else:
        # Model as two different lines for each band
        if mu <= 2 * t * d:
            guess_rho = mu / (4 * t * d) + 1 / 2
        elif 2 * t * d < mu <= U - 2 * t * d:
            guess_rho = 1
        elif U - 2 * t * d < mu:
            guess_rho = mu / (4 * t * d) + 1 - (U - 2 * t * d) / (4 * t * d)

    if U <= 4 * t * d:
        a = 0.1
        b = - a * (4 * t + U**2) / (4 * t + U)
        c = 2 * t * b - 4 * t**2 * a
        # Model as ax**2 + bx + c
        guess_e = a * mu**2 + b * mu + c
    else:
        a = 1 / 36
        # Model as two different parabolas for each band
        if mu <= 2 * t * d:
            guess_e = a * (mu**2 - 4 * (t * d)**2)
        elif 2 * t * d < mu <= U - 2 * t * d:
            guess_e = 0
        elif U - 2 * t * d < mu:
            b = - 2 * a * U 
            c = - a * (- U**2 + 4 * (t*d)**2)
            guess_e = a * mu**2 + b * mu + c


    guess = [guess_rho, guess_e]

    sol = root(GLS_reduced, guess, method='hybr')
    """
    # Robust handling:
    if not sol.success:
        # give useful debug info and raise to stop propagation of bad shapes
        raise RuntimeError(f"solve_GLS_2d_for_rho: inner solver failed for mu={mu}, U={U}: {sol.message}")
    """
    # Ensure we always return plain Python floats (no shape surprises)
    x = np.asarray(sol.x).ravel()      # now guaranteed shape (2,)
    rho_val = float(x[0])
    e_tilde = float(x[1])
    return rho_val, e_tilde


# Given an array of mu values, calculate correponding rho and e_tilde values, which are returned as separate arrays
# Uses direct 2d root finding
def create_solution_arrays_rho_e_root(mu_array: np.ndarray, U: float, f_0: float, f_1: float):
    rho_list = []
    e_tilde_list = []
    N = len(mu_array)
    i = 0

    for mu_val in mu_array:
        print(f'\rProgress: {(i / N * 100):.1f}%{' ' * 20}', end="", flush=True)
        sol = solve_GLS_2d_for_rho(mu_val, U, f_0, f_1)
        rho_list.append(sol[0])
        e_tilde_list.append(sol[1])
        i += 1
    print(f'\rProgress: {(i / N * 100):.1f}%{' ' * 20}', end="", flush=True)

    rho_array = np.array(rho_list)
    e_tilde_array = np.array(e_tilde_list)

    return rho_array, e_tilde_array
