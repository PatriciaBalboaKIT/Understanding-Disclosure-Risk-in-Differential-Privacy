import numpy as np

from functools import partial
from itertools import combinations
from scipy.optimize import minimize_scalar

from Bounds.f_functions import dp_sgd_objective

######## Bound variables #########
def compute_G(m):
    return (m-1)/m

def compute_optimization_bounds(eps, delta, m):
    lower = (1 - delta) / (m - 1 + np.exp(eps))
    upper = (1 - delta*np.exp(-eps)) / (m- 1 + np.exp(-eps))
    return lower, upper

######### Ours #####################
def theo_42(tv, kap):
    return tv * (1 - kap)

def theo_42_bb(kap, eps, delta):
    return ((np.exp(eps) - 1 +2*delta)/(np.exp(eps) + 1)) * (1 - kap)

def theo_51(epsilons, sigmas, delta, m, q, num_steps, data_dist):
    lower = min(data_dist)
    upper = max(data_dist)

    bounds = []
    for eps, sigma in zip(epsilons, sigmas):
        objective = partial(dp_sgd_objective, q=q, sigma=sigma, num_steps=num_steps)
        result = minimize_scalar(objective, bounds=(lower, upper), method='bounded')
        f_val = -result.fun  # max_alpha 1 - f(alpha) - alpha 
        bounds.append(f_val)

    return bounds

def theo_51_discrete(epsilons, sigmas, delta, m, q, num_steps, data_dist):
    kappa = sum(entry**2 for entry in data_dist)
    kappa_plus = max(data_dist)

    lower = 0
    upper = kappa_plus / (1-kappa)

    bounds = []
    for eps, sigma in zip(epsilons, sigmas):
        objective = partial(dp_sgd_objective, q=q, sigma=sigma, num_steps=num_steps)
        result = minimize_scalar(objective, bounds=(lower, upper), method='bounded')
        f_val = -result.fun  # max_alpha 1 - f(alpha) - alpha 
        bounds.append(f_val * (1-kappa))

    return bounds

def co_54(epsilons, sigmas, delta, m, q, num_steps):
    """
    Parameters
    ----------
    epsilons : list or array-like
        List of epsilon values.
    sigmas : list or array-like
        List of sigma values corresponding to epsilons.
    delta : float
        Privacy parameter delta.
    m : int
        Number of samples or partitions.
    q : float
        Sampling probability.
    num_steps : int
        Number of training steps.

    Returns
    -------
    list
        Computed DP-SGD bounds for each (epsilon, sigma).
    """
    bounds = []

    for eps, sigma in zip(epsilons, sigmas):
        lower, upper = compute_optimization_bounds(eps, delta, m)

        objective = partial(dp_sgd_objective, q=q, sigma=sigma, num_steps=num_steps)
        result = minimize_scalar(objective, bounds=(lower, upper), method='bounded')
        f_val = -result.fun  # max_alpha 1 - f(alpha) - alpha 

        bound = f_val * (m - 1) / m
        bounds.append(bound)

    return bounds