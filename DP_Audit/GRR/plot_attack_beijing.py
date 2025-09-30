import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

from Bounds.tv import tv_grr
from Geolife.constants import M

# file imports
main_dir = Path(__file__).parent.parent
data_file = main_dir / "GRR" /  "results" / "attack_results_Beijing.csv"

# Load the CSV file and extract attack results
df = pd.read_csv(data_file)
epsilons = df["Epsilon"][:20]
mean_rero = df["ReRo"][:20]
mean_u_rero = df["U-ReRo"][:20]


########### Plot U-ReRo + Bound ######################
# Compute bound: Example 5.4
dense_epsilons = np.arange(0,20,0.01)
bounds = []
for eps in dense_epsilons:
    tv = tv_grr(eps, M)
    bound = tv * (1 - 1/M)
    bounds.append(bound)


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