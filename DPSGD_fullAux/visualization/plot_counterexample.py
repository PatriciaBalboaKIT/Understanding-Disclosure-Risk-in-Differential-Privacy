import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from Bounds.bounds import theo_42
from Bounds.tv import tv_dp_sgd


##### Get experimental results
main_dir = Path(__file__).parent.parent
results_dir = main_dir / "results"
fig_dir = main_dir / "plots"

results_fashion = []
for file in results_dir.glob('mnist_eps*.csv'):
    df = pd.read_csv(file)
    results_fashion.append(df[["Eps", 'ReRo', 'U-ReRo','Hayes bound', 'Balle bound']])
combined_df_fashion = pd.concat(results_fashion, ignore_index=True)
combined_df_fashion.sort_values(by='Eps', inplace=True)

### Set parameters
delta = 1e-5
q = 1.0                             # sampling rate
C = 0.1                             # gradient clip (sensitivity)
num_steps = 100                     # training steps
epsilons = [0.512063674748734, 1.0125506277526433, 1.5567628736984664, 2.051797484856534, 5.0159472915120995, 10.7255096823]       # overall privacy budget
sigmas = [75, 40, 27, 21, 9.5, 5]         # noise multipliers
m = 8


results = combined_df_fashion

# aux bound
aux_bound = []
for sigma in sigmas:
    tv = tv_dp_sgd(sigma, num_steps)
    aux_bound.append(theo_42(tv,1/m))


# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(results['Eps'], results['ReRo'], label='ReRo', marker='o',color="C0", s=80)
plt.scatter(results['Eps'], results['U-ReRo'], label='RAD', marker='x',color="C1", s=110, zorder=10)
#plt.plot(results['Eps'], aux_bound, label="RAD bound Th.4.2", lw=3, color="#2ca02c")
plt.plot(results['Eps'], results['Hayes bound'], lw=3, label="ReRo bound [23]", color="C0")
plt.xlabel(r"Privacy budget $\varepsilon$", fontsize=20)
plt.xlim(0.4,2.2)
plt.ylim(-0.1,1.1)
plt.legend(fontsize=20)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.rcParams['pdf.fonttype'] = 42
plt.tight_layout()
plt.savefig(fig_dir / "DPSGD_fullAux_counterexample.png")