import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from Bounds.bounds import *
from Bounds.f_functions import  *
from Bounds.utils import compute_epsilons_and_sigmas
from Bounds.tv import *
from scipy.stats import norm



####### Parameters #########
m_val = 11                # number of attributes
q = 1.0                         # sampling rate
num_steps = 1                   # training steps
delta = 1e-5

############## Get eps,sigma pairs ########
sigmas, epsilons = compute_epsilons_and_sigmas(q, num_steps, delta)

######### General Bound TV ##########
"Requisites: "
"-TV(M)"
"-G(pi)"
"Z discrete"
gaussian_general_values = []
for e, sigma in zip(epsilons, sigmas):
    kap=(m_val-1)/m_val         # sensitivity of 1
    tv = tv_gaussian(sigma)
    gaussian_general_values.append(tv*kap)

laplace_general_values = []
for e in epsilons:
    kap=(m_val-1)/m_val         # sensitivity of 1
    tv = tv_laplace(e)
    laplace_general_values.append(tv*kap)

oue_general_values = []
for e in epsilons:
    kap=(m_val-1)/m_val         # sensitivity of 1
    tv = tv_oue(e)
    oue_general_values.append(tv*kap)


#### OPTIMAL BOUND
gaussian_rad_values=[]
d_vals = [0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1]  # m-1 valores
for e, sigma in zip(epsilons, sigmas):
    terms = norm.cdf(d_vals / (2 * sigma))
    bound_val = (2/m_val) * np.sum(terms) - (m_val-1)/m_val
    gaussian_rad_values.append(bound_val)


Laplace_rad_values=[]
for e in epsilons:
    bound= 1-np.exp(-e/(2*(m_val-1)))
    bound_val = bound*(m_val-1)/m_val
    Laplace_rad_values.append(bound_val)


oue_rad_values=[]
for e in epsilons:
    m=m_val
    p=np.exp(e)/(np.exp(e)+1)
    bound_val= (2*p-1)*(1-p**(m-1))/(2*m*(1-p))
    oue_rad_values.append(bound_val)

### PLOT
plt.figure(figsize=(10, 6))
#General Bount TH. 5.2
plt.plot(epsilons, gaussian_general_values, label="Gaussian Th. 4.2 ", linestyle="--", lw=4, color="#2731C1", zorder=15)
plt.plot(epsilons, laplace_general_values, label="Laplace Th. 4.2 ", linestyle="--", lw=4, color="#D69119", zorder=15)
plt.plot(epsilons, oue_general_values, label="OUE Th. 4.2 ", linestyle="--", lw=4, color="#BD2882", zorder=15)

plt.plot(epsilons, gaussian_rad_values, label="Gaussian Th. 4.3", linestyle="-", lw=4, color="#2731C1", zorder=15)
plt.plot(epsilons, Laplace_rad_values, label="Laplace Th. 4.3", linestyle="-", lw=4, color="#D69119", zorder=15)
plt.plot(epsilons, oue_rad_values, label="OUE Th. 4.3", linestyle="-", lw=4, color="#BD2882", zorder=15)

plt.xlabel(r'Privacy budget $\varepsilon$', fontsize=20)
plt.ylabel(r'$0$-RAD', fontsize=20)
plt.ylim(-0.1, 1.1)
plt.xlim(-0.1,10)
plt.legend(fontsize=20)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5),
           fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.tick_params(axis='both', which='major', labelsize=18)
plt.rcParams['pdf.fonttype'] = 42
plt.savefig(f"./Bounds/plots/compare_theorems.png", dpi=300, bbox_inches="tight")
plt.show()