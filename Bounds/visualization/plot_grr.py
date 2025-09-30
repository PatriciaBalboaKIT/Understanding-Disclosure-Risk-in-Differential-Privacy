import numpy as np
import matplotlib.pyplot as plt
from math import comb
from scipy.optimize import bisect, brentq

# ----------------------
# CONFIGURATION
# ----------------------
n = 1000                # Data size
beta_target = 0.05     # Target beta
theta = 1              # Proportion of 1s in the true data
m = 10
kappa = 1 / m


# ----------------------
# GRR f-function
# ----------------------
def f_grr(x, eps):
    """f-function for Generalized Randomized Response (GRR)."""
    q = 1 / (np.exp(eps) + m - 1)
    p = np.exp(eps) / (np.exp(eps) + m - 1)

    if x <= q:
        return 1 - np.exp(eps) * x
    elif x <= (1 - p):
        return m * q - x
    else:
        return np.exp(-eps) * (1 - x)


# ----------------------
# Hayes ReRo bound
# ----------------------
def g(eps, kappa):
    return 1 - f_grr(eps, kappa)

def g_inverse(y, kappa):
    func = lambda eps: g(eps, kappa) - y
    try:
        return brentq(func, 1e-6, 20)
    except ValueError:
        return None

# ----------------------
# Thm. 5.3 bound
# ----------------------
def h(eps, m):
    """h(eps) = (m-1)/m * (1 - exp(-eps/(2(m-1)))) (simplified form)."""
    return (1 - kappa) * (np.exp(eps) - 1) / (np.exp(eps) + m - 1)


def h_inverse(y, m):
    """Inverse of h using root finding."""
    func = lambda eps: h(eps, m) - y
    try:
        return brentq(func, 1e-6, 50)
    except ValueError:
        return None


# ----------------------
# AUXILIARY FUNCTIONS
# ----------------------
def binom_pmf(k, n, q):
    """Binomial PMF."""
    return comb(n, k) * (q**k) * ((1 - q) ** (n - k))


def binom_tail(t, n, q):
    """Tail probability: P(K >= t)."""
    if t > n:
        return 0.0
    t = max(t, 0)
    return sum(binom_pmf(k, n, q) for k in range(t, n + 1))


def compute_beta_rr(alpha, q, scale, n):
    """Compute probability of error beta given alpha."""
    t = int(np.ceil(alpha * (1 / scale)))
    return binom_tail(t, n, q)


def alpha(beta, q, scale, n, alpha_max=1000, tol=1e-5):
    """Compute alpha given beta by root finding."""
    def f(a):
        return compute_beta_rr(a, q, scale, n) - beta
    return bisect(f, 0, alpha_max, xtol=tol)


# ----------------------
# MAIN
# ----------------------
xs = np.linspace(0.001, 0.8, 10)
alphas_h = []
alphas_g = []
scale_rr = abs(1 - 2 * theta)
for x in xs:
    print("Calculating risk=",x)
    # ReRo bound (g)
    eps_g = g_inverse(x, kappa)
    print("ReRo eps=",eps_g)
    if eps_g is not None:
        q_g = 1 / (np.exp(eps_g) + m - 1)
        alphas_g.append(alpha(beta_target, q_g, scale_rr, n))  
    else:
        alphas_g.append(np.nan)
    # RAD bound (h)
    eps_h = h_inverse(x, m)
    print("RAD eps=",eps_h)
    if eps_h is not None:
        q_h = 1 / (np.exp(eps_h) + m - 1)
        alphas_h.append(alpha(beta_target, q_h, scale_rr, n))
    else:
        alphas_h.append(np.nan)


# ----------------------
# PLOT
# ----------------------
plt.plot(xs, alphas_g, label=r"ReRo bound [23]", lw=4, color="C0")
plt.plot(xs, alphas_h, label=r"RAD bound Th.4.3", lw=4, color="#2ca02c")
plt.yscale("log")
plt.xlabel("Accepted risk", fontsize=20)
plt.ylabel(r"Error $\alpha$", fontsize=20)
plt.legend(fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=18)
plt.grid(True)
plt.rcParams['pdf.fonttype'] = 42
plt.savefig(f"./Bounds/plots/GRR_accu_m{m}.png", dpi=300, bbox_inches="tight")
plt.show()
