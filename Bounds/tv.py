import numpy as np
from scipy.stats import multivariate_normal, norm

def tv_dp_sgd(noise_mul, steps, mc_samples=10000):
    """
    Compute TV(μ, ν) where:
    - ν = N(0, σ² I)
    - μ = N(1_T, σ² I), because q = 1 ⇒ deterministic w = 1_T
    """
    T = steps
    sigma = noise_mul

    # Sample from ν = N(0, σ^2 I)
    samples = np.random.normal(loc=0, scale=sigma, size=(mc_samples, T))

    # Evaluate densities
    nu = multivariate_normal(mean=np.zeros(T), cov=sigma**2 * np.eye(T))
    mu = multivariate_normal(mean=np.ones(T), cov=sigma**2 * np.eye(T))

    nu_vals = nu.pdf(samples)
    mu_vals = mu.pdf(samples)

    # TV estimate
    ratio = mu_vals / nu_vals
    indicator = mu_vals <= nu_vals
    tv_estimate = np.mean((1 - ratio) * indicator)

    return tv_estimate


def tv_gaussian(sigma):
    mu=1/sigma              # sensitivity of 1
    x=norm.cdf(mu/2)
    tv=2*x-1                # TV for 1 step
    return tv

def tv_geometric_mech(x, y, q):
    k = abs(x - y)
    if k % 2 == 0: 
        tv = 1 - q**(k // 2)
    else:
        tv = 1 - 2 * q**((k+1)//2) / (1 + q)
    return tv

def tv_grr(eps, m):
    tv = (np.exp(eps) -1) / (np.exp(eps) + m - 1)
    return tv

def tv_laplace(eps):
    return 1 - np.exp(-eps/2)  

def tv_oue(eps):
    tv = 1/2 * (np.exp(eps) -1) / (np.exp(eps) + 1)
    return tv