import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

from Bounds.tv import tv_grr
from Porto.constants import M

# file imports
main_dir = Path(__file__).parent.parent
data_file_1 = main_dir / "GRR" / "results" / "attack_results.csv"
data_file_2 = main_dir / "SS" / "results" / "attack_results.csv"
data_file_3 = main_dir / "UE" / "results" / "attack_results_oue.csv"
# Load the CSV file and extract attack results
df = pd.read_csv(data_file_1)
epsilons_GRR = df["Epsilon"][:15]
mean_rero_GRR = df["ReRo"][:15]
mean_u_rero_GRR = df["U-ReRo"][:15]

df = pd.read_csv(data_file_2)
epsilons_SS = df["Epsilon"][:15]
mean_rero_SS = df["ReRo"][:15]
mean_u_rero_SS = df["U-ReRo"][:15]

df = pd.read_csv(data_file_3)
epsilons_OUE = df["Epsilon"][:15]
mean_rero_OUE = df["ReRo"][:15]
mean_u_rero_OUE = df["U-ReRo"][:15]


########### Plot U-ReRo + Bound ######################
# Compute bound: Example 5.4
dense_epsilons = np.arange(0,15,0.01)
bounds_grr = []
for eps in dense_epsilons:
    tv = tv_grr(eps, M)
    bound = tv * (1 - 1/M)
    bounds_grr.append(bound)

########### Plot U-ReRo + Bound ######################
bounds_oue = []
for eps in dense_epsilons:
    q = 1/(np.exp(eps) + 1)
    p = 1 - q
    numerator = (2*p -1) * (1-p**(M-1))
    denominator = 2*M*(1-p)
    value = numerator / denominator
    bounds_oue.append(value)

bounds_ss = []
for eps in dense_epsilons:
    w = max(1, np.floor(M / (np.exp(eps) + 1)))
    p = (w * np.exp(eps))/(w*np.exp(eps) + M - w)
    value = (p * M - w) / (M * w)
    bounds_ss.append(value)


# Create the plot
plt.figure(figsize=(9, 6))
plt.scatter(epsilons_OUE, mean_u_rero_OUE, label="RAD OUE", marker="o", color="#BD2882", s=100, zorder=20)
plt.scatter(epsilons_GRR, mean_u_rero_GRR, label="RAD GRR", marker="s", color="#235288", s=100, zorder=20)
plt.scatter(epsilons_SS, mean_u_rero_SS, label="RAD SS", marker="v", color="#23cb34", s=80, zorder=20)
plt.plot(dense_epsilons, bounds_grr, label="GRR Th.4.3/\nBlack-box Co.5.4", linestyle="-", lw=4, color="#235288")
plt.plot(dense_epsilons, bounds_oue, label="OUE Th.4.3", linestyle="-", lw=4, color="#BD2882")
plt.plot(dense_epsilons, bounds_ss, label="SS Th.4.3", linestyle="-", lw=2, color="#23cb34")
plt.xlabel(r"Privacy budget $\varepsilon$", fontsize=20)
plt.ylabel(r'$0$-RAD', fontsize=20)
plt.ylim(-0.1,1.1)
plt.xlim(2,13)
plt.legend(fontsize=20,loc="upper left",bbox_to_anchor=(0,1))
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tight_layout()
plt.rcParams['pdf.fonttype'] = 42
plt.savefig("./DP_Audit/plots/LDP_attacks_porto.png")
plt.show()