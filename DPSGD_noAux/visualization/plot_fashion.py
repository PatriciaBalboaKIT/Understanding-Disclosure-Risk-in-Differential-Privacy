import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from Bounds.bounds import theo_51_discrete


##### Get experimental results: uniform prior
main_dir = Path(__file__).parent.parent
results_dir = main_dir / "results"
fig_dir = main_dir / "plots"

results_mnist = []
for file in results_dir.glob('fashion*.csv'):
    df = pd.read_csv(file)
    results_mnist.append(df[['Eps', 'ReRo', 'U-ReRo', 'Hayes bound']])
combined_df_mnist = pd.concat(results_mnist, ignore_index=True)
combined_df_mnist.sort_values(by='Eps', inplace=True)

### Set parameters
delta = 1e-5
q = 1.0                             # sampling rate
C = 0.1                             # gradient clip (sensitivity)
num_steps = 100                     # training steps
epsilons = [0.512063674748734, 1.0125506277526433, 1.5567628736984664, 2.051797484856534, 5.0159472915120995, 10.725509682302643]       # overall privacy budget
sigmas = [75, 40,  27, 21, 9.5, 5]         # noise multipliers
m = 8
data_dist =  m * [1/m]               # uniform distribution

results = combined_df_mnist

# Compute bounds
bounds = theo_51_discrete(epsilons, sigmas,delta, m, q, num_steps, data_dist)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(results['Eps'], results['ReRo'], label='ReRo', marker='o',color="C0", s=80)
plt.scatter(results['Eps'], results['U-ReRo'], label='RAD', marker='x',color="C1", s=110, zorder=10)
plt.plot(epsilons, bounds, label="RAD bound Th.5.1", lw=3, color="#2ca02c")
plt.plot(results['Eps'], results['Hayes bound'], lw=3, label="ReRo bound [23]", color="C0")
plt.xlabel(r"Privacy budget $\varepsilon$", fontsize=20)
plt.xlim(0,11)
plt.ylim(-0.1,1.1)
plt.legend(fontsize=20)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.rcParams['pdf.fonttype'] = 42
plt.tight_layout()
plt.savefig(fig_dir / f"Fashion_m={m}.png")
