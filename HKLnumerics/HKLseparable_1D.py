# NOTE:
# This file contains many functions. This is because the numerics of this HKL model are somwhat unstable and different 
# Ansatzes were implemented. There's generally two Ansatzes containing two different Methods each. They are as follows:
#
# 1. Take rho array, calculate corresponding mu values
#   Method 1: Direct root finding of 2d function defined by the system of equations for rho and e_tilde
#   Method 2: As the above seems to be highly unstable, try to find minimum in rho and e_tilde in the absolute norm of said
#             function.
# 2. Take mu array, calculate corresponding rho values. For this, again a direct root finding method and a minimization
#    of the abolute norm is implemented.
#
# It seems that the best compromise between stability and efficiency is direct root finding wrt rho, so Ansatz 2.1.

import numpy as np
from scipy.optimize import root, minimize

t = 1
d = 1

def create_mu_array(N: int, U: float, f_0: float):
    return np.linspace(-2 * t, 2 * t + U + 2 * f_0, N)



# =======
# DENSITY
# =======

# I_1 with two variables
def I_1(x: float, y: float):
    if np.abs(x / (2 * t * y)) < 1:
        return np.heaviside(x + 2 * t * y, 1) - np.arccos(x / (2 * t * y)) / np.pi
    else:
        return np.heaviside(x + 2 * t * y, 1)
    
# J_1 with two varibales
def J_1(x: float, y: float):
    if np.abs(x / (2 * t * y)) < 1:
        return - np.sqrt((2 * t * y)**2 - x**2) / np.pi
    else:
        return 0


# The non-linear system of equations that needs to be solved
# Only for U > 0!   
def GLS_1d(rho: float, mu: float, U: float, e_tilde: float, f_0: float, f_1: float):
    eq1 = rho - (I_1(mu - f_0 * rho, 1 + f_1 * e_tilde) + I_1(mu - U - f_0 * rho, 1 + f_1 * e_tilde))
    eq2 = e_tilde - (J_1(mu - f_0 * rho, 1 + f_1 * e_tilde) + J_1(mu - U - f_0 * rho, 1 + f_1 * e_tilde))
    return [eq1, eq2]


#---- Ansatz 1.1 --------

# Direct 2d root finding
def solve_GLS_1d_for_mu(rho: float, U: float, f_0: float, f_1: float, initial_guess: list):
    # def x = [mu, e_tilde]
    GLS_reduced = lambda x: GLS_1d(rho, x[0], U, x[1], f_0, f_1)
    sol = root(GLS_reduced, initial_guess, method='hybr')
    # Should return list [mu, e_tilde] for any given rho
    return sol.x


# Given an array of rho values, solve the SOE for mu and e_tilde, return two arrays with the corresponding values
# Uses direct root finding method
def create_solution_arrays_mu_e_root(rho_array: np.ndarray, U: float, f_0: float, f_1: float):
    mu_list = []
    e_tilde_list = []

    for rho_val in rho_array:
        if U <= 4 * t * d:
            # Model as one straight line through start and endpoint
            guess_rho = (U / 2 + 2) * rho_val - 2
        else:
            # Model as two different lines for each band
            if rho_val <= 1:
                guess_rho = 4 * rho_val - 2
            else:
                guess_rho = 4 * rho_val + U - 6

        a = 1

        if U <= 4 * t * d:
            # Model as a(x**2 - 2x)
            guess_e = a * (rho_val**2 - 2 * rho_val)
        else:
            # Model as two different parabolas for each band
            if rho_val <= 1:
                guess_e = a * (rho_val**2 - rho_val)
            else:
                guess_e = a * (rho_val**2 - 3 * rho_val + 2)


        guess = [guess_rho, guess_e]

        sol = solve_GLS_1d_for_mu(rho_val, U, f_0, f_1, guess)
        mu_list.append(sol[0])
        e_tilde_list.append(sol[1])

    mu_array = np.array(mu_list)
    e_tilde_array = np.array(e_tilde_list)

    return mu_array, e_tilde_array


#---- Ansatz 1.2 --------

# Instead of direct root finding, search for minimum in the norm of the function defined by SOE
def minimize_GLS_norm_1d_wrt_mu(rho: float, U: float, f_0: float, f_1: float, initial_guess: list):
    # def x = [mu, e_tilde]
    GLS_reduced = lambda x: GLS_1d(rho, x[0], U, x[1], f_0, f_1)
    GLS_reduced_norm = lambda x: GLS_reduced(x)[0]**2 + GLS_reduced(x)[1]**2

    sol = minimize(GLS_reduced_norm, initial_guess, method='L-BFGS-B')
    # Should return list [mu, e_tilde] for any given rho
    return sol.x


# Given an array of rho values, solve the SOE for mu and e_tilde, return two arrays with the corresponding values
# Uses minimization of norm
def create_solution_arrays_mu_e_norm(rho_array: np.ndarray, U: float, f_0: float, f_1: float):
    mu_list = []
    e_tilde_list = []

    for rho_val in rho_array:
        if U <= 4 * t * d:
            # Model as one straight line through start and endpoint
            guess_rho = (U / 2 + 2) * rho_val - 2
        else:
            # Model as two different lines for each band
            if rho_val <= 1:
                guess_rho = 4 * rho_val - 2
            else:
                guess_rho = 4 * rho_val + U - 6

        a = 1

        if U <= 4 * t * d:
            # Model as a(x**2 - 2x)
            guess_e = a * (rho_val**2 - 2 * rho_val)
        else:
            # Model as two different parabolas for each band
            if rho_val <= 1:
                guess_e = a * (rho_val**2 - rho_val)
            else:
                guess_e = a * (rho_val**2 - 3 * rho_val + 2)


        guess = [guess_rho, guess_e]

        sol = minimize_GLS_norm_1d_wrt_mu(rho_val, U, f_0, f_1, guess)
        mu_list.append(sol[0])
        e_tilde_list.append(sol[1])

    mu_array = np.array(mu_list)
    e_tilde_array = np.array(e_tilde_list)

    return mu_array, e_tilde_array




# Idea: Dont take rho array as given, but mu array. compute rho(mu). As this should not have points where the
# derivative is near zero, newton-techniques will likely be more succesfull and we can get rid of kinks in 
# numerical calculations.

#---- Ansatz 2.1 --------

# Do not solve SOE for mu, but for rho
def solve_GLS_1d_for_rho(mu: float, U: float, f_0: float, f_1: float):
    # def x = [rho, e_tilde]
    GLS_reduced = lambda x: GLS_1d(x[0], mu, U, x[1], f_0, f_1)

    # Guess something near a possible solution
    if U <= 4 * t * d:
        # Model as one straight line through start and endpoint
        guess_rho = (2 / (4 * t + U)) * mu + 4 * t / (4 * t + U)
    else:
        # Model as two different lines for each band
        if mu <= 2 * t:
            guess_rho = mu / (4 * t) + 1 / (2 * t)
        elif 2 * t < mu <= U - 2 * t:
            guess_rho = 1
        elif U - 2 * t < mu:
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

    guess = [guess_rho, 0]

    sol = root(GLS_reduced, guess, method='hybr')
    # Should return list [rho, e_tilde] for any given mu
    return sol.x


# Given an array of mu values, calculate correponding rho and e_tilde values, which are returned as separate arrays
# Uses direct 2d root finding
def create_solution_arrays_rho_e_root(mu_array: np.ndarray, U: float, f_0: float, f_1: float):
    rho_list = []
    e_tilde_list = []

    for mu_val in mu_array:
        sol = solve_GLS_1d_for_rho(mu_val, U, f_0, f_1)
        rho_list.append(sol[0])
        e_tilde_list.append(sol[1])

    rho_array = np.array(rho_list)
    e_tilde_array = np.array(e_tilde_list)

    return rho_array, e_tilde_array


#---- Ansatz 2.2 --------

# Instead of direct root finding, search for minimum in the norm of the function defined by SOE
def minimize_GLS_norm_1d_rho_e(mu: float, U: float, f_0: float, f_1: float, initial_guess: list):
    # def x = [rho, e_tilde]
    GLS_reduced = lambda x: GLS_1d(x[0], mu, U, x[1], f_0, f_1)
    GLS_reduced_norm = lambda x: GLS_reduced(x)[0]**2 + GLS_reduced(x)[1]**2

    sol = minimize(GLS_reduced_norm, initial_guess, method='L-BFGS-B')
    # Should return list [rho, e_tilde] for any given rho
    return sol.x


# Given an array of mu values, calculate correponding rho and e_tilde values, which are returned as separate arrays
# Uses minimization of norm
def make_solution_arrays_rho_e_norm(mu_array: np.ndarray, U: float, f_0: float, f_1: float):
    rho_list = []
    e_tilde_list = []

    for mu_val in mu_array:
        if U <= 4 * t * d:
            # Model as one straight line through start and endpoint
            guess_rho = (2 / (4 * t + U)) * mu_val + 4 * t / (4 * t + U)
        else:
            # Model as two different lines for each band
            if mu_val <= 2 * t:
                guess_rho = mu_val / (4 * t) + 1 / (2 * t)
            elif 2 * t < mu_val <= U - 2 * t:
                guess_rho = 1
            elif U - 2 * t < mu_val:
                guess_rho = mu_val / (4 * t) + (3 - U / (2 * t)) / 2


        if U <= 4 * t * d:
            a = 1
            b = - a * (4 * t + U**2) / (4 * t + U)
            c = 2 * t * b - 4 * t**2 * a
            # Model as ax**2 + bx + c
            guess_e = a * mu_val**2 + b * mu_val + c
        else:
            a = 0.5
            # Model as two different parabolas for each band
            if mu_val <= 2 * t:
                guess_e = a * (mu_val**2 - 4 * t**2 * a)
            elif 2 * t < mu_val <= U - 2 * t:
                guess_e = 0
            elif U - 2 * t < mu_val:
                b = - a *  (4 * t * U) / (U + 2 * t)
                c = - (U**2 + 2 * t**2) * a - U * b
                guess_e = a * mu_val**2 + b * mu_val + c


        guess = [guess_rho, 0]

        sol = minimize_GLS_norm_1d_rho_e(mu_val, U, f_0, f_1, guess)
        rho_list.append(sol[0])
        e_tilde_list.append(sol[1])

    rho_array = np.array(rho_list)
    e_tilde_array = np.array(e_tilde_list)

    return rho_array, e_tilde_array






# ==============
# ENERGY DENSITY
# ==============


def create_energy_array(mu_array, rho_array, e_array, U, f_0, f_1):
    energy_list = []
    N = len(mu_array)
    i = 0

    for i in range(N):
        rho_val, mu_val = rho_array[i], mu_array[i]
        print(f'\rProgress: {(i / N * 100):.1f}%{' ' * 20}', end="", flush=True)
        e_val = e_array[i] + U * I_1(mu_val - U - f_0 * rho_val, 1 + f_1 * e_array[i])
        energy_list.append(e_val)
        i += 1

    energy_array = np.array(energy_list)

    return energy_array
