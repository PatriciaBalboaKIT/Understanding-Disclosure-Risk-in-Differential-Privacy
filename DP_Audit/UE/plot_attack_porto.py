import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

from Porto.constants import M

# file imports
main_dir = Path(__file__).parent.parent
data_file = main_dir / "UE" /  "results" / "attack_results_oue.csv"

# Load the CSV file and extract attack results
df = pd.read_csv(data_file)
epsilons = df["Epsilon"][:20]
mean_rero = df["ReRo"][:20]
mean_u_rero = df["U-ReRo"][:20]


########### Plot U-ReRo + Bound ######################
# Compute bound: Example 5.5
dense_epsilons = np.arange(1,20,0.001)
bounds = []
for eps in dense_epsilons:
    q = 1/(np.exp(eps) + 1)
    p = 1 - q
    numerator = (2*p -1) * (1-p**(M-1))
    denominator = 2*M*(1-p)
    value = numerator / denominator
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