import numpy as np
from scipy.optimize import root

t = 1
d = 3


def create_mu_array(N: int, U: float, f_0: float):
    return np.linspace(-2 * t * d, 2 * t * d + U + 2 * f_0, N)


N = 1000

# np.where(if statement, if true, if false)
def I_1(x: np.ndarray, y: float):
    return np.where(np.abs(x) < (2 * t * y),
             np.heaviside(x + 2*t*y, 1) - (1 / np.pi) * np.arccos(np.clip(x / (2*t*y), -1, 1)),
             np.heaviside(x + 2*t*y, 1))

k1_array = np.linspace(-np.pi, np.pi, N)
k2_array = np.linspace(-np.pi, np.pi, N)
k1k1, k2k2 = np.meshgrid(k1_array, k2_array)

def I_3(x: float, y: float):
    shifted_I_1 = x + 2 * t * y * (np.cos(k1k1) + np.cos(k2k2))
    func_val_grid = I_1(shifted_I_1, y)

    intermediate = np.trapezoid(func_val_grid, k2_array, axis=1)
    result = np.trapezoid(intermediate, k1_array)

    return result / ((2 * np.pi)**2)


# np.where(if statement, if true, if false)
def J_1(x: np.ndarray, y: float):
    return np.where(np.abs(x) < (2 * t * y),
             - 1 / (np.pi * np.sqrt(np.square(2 * t * y) - np.square(x))),
             0)


def J_3(x: float, y: float):
    shifted_I_1 = x + 2 * t * y * (np.cos(k1k1) + np.cos(k2k2))
    func_val_grid = J_1(shifted_I_1, y)

    intermediate = np.trapezoid(func_val_grid, k2_array, axis=1)
    result = np.trapezoid(intermediate, k1_array)

    return d * result / ((2 * np.pi)**2)


def GLS_3d(rho: float, mu: float, U: float, e_tilde: float, f_0: float, f_1: float):
    eq1 = rho - (I_3(mu - f_0 * rho, 1 + f_1 * e_tilde) + I_3(mu - U - f_0 * rho, 1 + f_1 * e_tilde))
    eq2 = e_tilde - (J_3(mu - f_0 * rho, 1 + f_1 * e_tilde) + J_3(mu - U - f_0 * rho, 1 + f_1 * e_tilde))
    return [eq1, eq2]

# Do not solve SOE for mu, but for rho
def solve_GLS_3d_for_rho(mu: float, U: float, f_0: float, f_1: float):
    # def x = [rho, e_tilde]
    GLS_reduced = lambda x: GLS_3d(x[0], mu, U, x[1], f_0, f_1)

    # Guess something near a possible solution
    if U <= 4 * t * d:
        # Model as one straight line through start and endpoint
        guess_rho = (2 / (4 * t + U)) * mu + 4 * t / (4 * t + U)
    else:
        # Model as two different lines for each band
        if mu <= 2 * t * d:
            guess_rho = mu / (4 * t) + 1 / (2 * t)
        elif 2 * t * d < mu <= U - 2 * t * d:
            guess_rho = 1
        elif U - 2 * t * d < mu:
            guess_rho = mu / (4 * t) + (3 - U / (2 * t)) / 2

    if U <= 4 * t * d:
        a = 1
        b = - a * (4 * t + U**2) / (4 * t + U)
        c = 2 * t * b - 4 * t**2 * a
        # Model as ax**2 + bx + c
        guess_e = a * mu**2 + b * mu + c
    else:
        a = 0.5
        # Model as two different parabolas for each band
        if mu <= 2 * t:
            guess_e = a * (mu**2 - 4 * t**2 * a)
        elif 2 * t < mu <= U - 2 * t:
            guess_e = 0
        elif U - 2 * t < mu:
            b = - a *  (4 * t * U) / (U + 2 * t)
            c = - (U**2 + 2 * t**2) * a - U * b
            guess_e = a * mu**2 + b * mu + c

    guess = [guess_rho, guess_e]

    sol = root(GLS_reduced, guess, method='hybr', tol=1e-3)
    # Should return list [rho, e_tilde] for any given mu
    return sol.x


# Given an array of mu values, calculate correponding rho and e_tilde values, which are returned as separate arrays
# Uses direct 2d root finding
def create_solution_arrays_rho_e_root(mu_array: np.ndarray, U: float, f_0: float, f_1: float):
    rho_list = []
    e_tilde_list = []
    N = len(mu_array)
    i = 0

    for mu_val in mu_array:
        print(f'\rProgress: {(i / N * 100):.1f}%{' ' * 20}', end="", flush=True)
        sol = solve_GLS_3d_for_rho(mu_val, U, f_0, f_1)
        rho_list.append(sol[0])
        e_tilde_list.append(sol[1])
        i += 1
    print(f'\rProgress: {(i / N * 100):.1f}%{' ' * 20}', end="", flush=True)

    rho_array = np.array(rho_list)
    e_tilde_array = np.array(e_tilde_list)

    return rho_array, e_tilde_array