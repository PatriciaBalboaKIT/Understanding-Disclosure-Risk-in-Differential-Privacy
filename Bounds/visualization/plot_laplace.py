import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from Bounds.f_functions import f_laplace
from Bounds.tv import tv_laplace

# Parameters
beta = 1-0.95
m = 10

## Laplace mech Accuracy formula
def alpha(eps):
    return np.log(1/beta) / eps


## Lapalce f-function
def f_eps_x(eps, x):
    if x < np.exp(-eps)/2:
        return 1 - np.exp(eps) * x
    elif np.exp(-eps)/2 <= x <= 0.5:
        return np.exp(-eps) / (4*x)
    else:
        return np.exp(-eps) * (1 - x)

#HAYES ReRo bound
# g(eps) = 1 - f_eps(1/m)
def g(eps, m):
    return 1 - f_laplace(1/m, eps)

def g_inverse(y, m):
    func = lambda eps: g(eps, m) - y
    try:
        return brentq(func, 1e-6, 20)
    except ValueError:
        return None

#TH 5.3 BOUND
# h(eps) = (m-1)/m * (1 - exp(-eps/(2(m-1))))
def h(eps, m):
    return (m-1)/m * (1 - np.exp(-eps/(2*(m-1))))

def h_inverse(y, m):
    func = lambda eps: h(eps, m) - y
    try:
        return brentq(func, 1e-6, 50)
    except ValueError:
        return None


#TH 5.2 BOUND
def k(eps,m):
    return tv_laplace(eps)*(1-1/m)
def k_inverse(y, m):
    func = lambda eps: k(eps, m) - y
    try:
        return brentq(func, 1e-6, 50)
    except ValueError:
        return None
    

xs = np.linspace(0.001, 0.8, 300)

alphas_g = []
alphas_h = []
alphas_k = []

for x in xs:
    # curva original con g
    eps_g = g_inverse(x, m)
    if eps_g is not None:
        alphas_g.append(alpha(eps_g))
    else:
        alphas_g.append(np.nan)
    
    # curva calibrada con h
    eps_h = h_inverse(x, m)
    if eps_h is not None:
        alphas_h.append(alpha(eps_h))
    else:
        alphas_h.append(np.nan)

    # curva calibrada con k
    eps_k = k_inverse(x, m)
    if eps_k is not None:
        alphas_k.append(alpha(eps_k))
    else:
        alphas_k.append(np.nan)

plt.plot(xs, alphas_g, label=fr"ReRo bound [23]",lw=4, color="C0")
plt.plot(xs, alphas_h, label=fr"RAD bound Th.4.3", lw=4, color="#2ca02c",)
plt.yscale("log")   # eje y logarítmico (base e)
plt.xlabel("Accepted risk",fontsize=20)
plt.ylabel(r"Error $\alpha$",fontsize=20)
plt.legend(fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.grid(True)
plt.rcParams['pdf.fonttype'] = 42
plt.savefig(f"./Bounds/plots/Laplace_accu_m{m}.png", dpi=300, bbox_inches="tight")