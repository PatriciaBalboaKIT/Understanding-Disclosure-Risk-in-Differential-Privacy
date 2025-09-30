import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.special import comb

from Porto.constants import M

# file imports
main_dir = Path(__file__).parent.parent
data_file = main_dir / "SS" /  "results" / "attack_results.csv"

# Load the CSV file and extract attack results
df = pd.read_csv(data_file)
epsilons = df["Epsilon"][:20]
mean_rero = df["ReRo"][:20]
mean_u_rero = df["U-ReRo"][:20]


########### Plot U-ReRo + Bound ######################
# Compute bound: Example 5.6
dense_epsilons = np.arange(1,20,0.001)
bounds = []
for eps in dense_epsilons:
    w = max(1, np.floor(M / (np.exp(eps) + 1)))
    p = (w * np.exp(eps))/(w*np.exp(eps) + M - w)
    value = (p * M - w) / (M * w)
    bounds.append(value)


# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(epsilons, mean_u_rero, label="RAD", marker="x", color="#ff7f0e", s=80, zorder=10)
plt.plot(dense_epsilons, bounds, label="RAD bound (Th.5.3)", linestyle="-", lw=3, color="#2ca02c")
plt.xlabel(r"Privacy budget $\varepsilon$", fontsize=18)
plt.ylim(-0.1,1.1)
plt.xlim(0,20)
plt.legend(fontsize=16)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.show()