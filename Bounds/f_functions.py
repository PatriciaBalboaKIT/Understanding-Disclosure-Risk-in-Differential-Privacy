import numpy as np
from scipy.stats import norm

def f_black_box(alpha, eps, delta):
    term1 = 1 - delta - np.exp(eps) * alpha
    term2 = np.exp(-eps) * (1 - delta - alpha)
    return max(0, term1, term2)

# Based on Hayes et al.: this corresponds to 1 - f(alpha)
def f_dp_sgd_approx(alpha, q, noise_mul, steps, mc_samples=10000):
    x = np.random.normal(0.0, noise_mul, (mc_samples,steps))
    per_step_log_ratio= np.log(1-q + q*(np.exp((-(x-1.0)**2 + (x)**2)/(2*noise_mul**2))))
    log_ratio=np.sum(per_step_log_ratio,axis=1)
    log_ratio=np.sort(log_ratio)
    r=np.exp(log_ratio)
    upper_bound=max(0.0,1-(1-alpha)*np.mean(r[:int(mc_samples*(1-alpha))]))
    return min(1.0, upper_bound)

def dp_sgd_objective(alpha, q, sigma, num_steps):
    # 1 - f(alpha) - alpha
    return -(f_dp_sgd_approx(alpha, q, sigma, num_steps) - alpha)


def f_gaussian(alpha, mu):
    return norm.cdf(norm.ppf(1 - alpha) - mu)

def f_grr(alpha, eps, p, q, m):
    if alpha <= q:
        return 1 - np.exp(eps) * alpha
    elif alpha <= (1 - p):
        return m * q - alpha
    else:
        return np.exp(-eps) * (1 - alpha)

def f_laplace(alpha, eps):
    if alpha < np.exp(-eps) / 2:
        return 1 - np.exp(eps) * alpha
    elif alpha <= 1/2:
        return np.exp(-eps) / (4 * alpha)
    else:
        return np.exp(-eps) * (1 - alpha)
    
def f_oue(alpha, eps, q):
    if alpha <= (q/2):
        return 1 - np.exp(eps) * alpha
    elif alpha <= ((1 + q)/ 2):
        return (1/2 + q) - alpha
    else:
        return np.exp(-eps) * (1 - alpha)
    
def f_ss(alpha, eps):
    term = np.exp(eps) * alpha / (1+(np.exp(eps) - 1)*alpha)
    return min(1, max(0, term))