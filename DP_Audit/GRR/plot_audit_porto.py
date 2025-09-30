import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# === File paths ===
our_results_csv = "DP_Audit/GRR/results/eps_estimation.csv"        # your computed results
ldp_auditor_csv = "DP_Audit/LDP_Auditor/results/summary_Porto.csv"  # LDP Auditor results
plot_path = "DP_Audit/GRR/plots/EpsEstimation.png"

# === Load our results ===
df_ours = pd.read_csv(our_results_csv)
epsilons = df_ours["eps"]
empirical_epsilons = df_ours["empirical_eps"]
empirical_epsilons_std = df_ours["std"]

# === Load LDP Auditor results ===
df_comp = pd.read_csv(ldp_auditor_csv)
df_comp_filtered = df_comp[df_comp["protocol"] == "GRR"].sort_values(by="epsilon")

epsilons_comp = df_comp_filtered["epsilon"][:len(epsilons)]
empirical_eps_comp = df_comp_filtered["avg_eps_emp"][:len(epsilons)]
std_comp = df_comp_filtered["std_eps_emp"][:len(epsilons)]

# === Plot ===
plt.figure(figsize=(8, 6))
plt.errorbar(
    epsilons,
    empirical_epsilons,
    yerr=empirical_epsilons_std,
    fmt='x',
    ms=10,
    mew=2,
    color='red',
    capsize=5,
    label="Ours",
    zorder=10
)
plt.errorbar(
    epsilons_comp,
    empirical_eps_comp,
    yerr=std_comp,
    fmt='o',
    ms=10,
    color='blue',
    capsize=5,
    label="LDP Auditor [4]"
)
plt.plot(epsilons, epsilons, color='gray', linestyle="--", label="Expectation")
plt.xlabel(r"Theoretical $\varepsilon$", fontsize=20)
plt.ylabel(r"Empirical $\varepsilon$", fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tight_layout()
plt.rcParams['pdf.fonttype'] = 42
plt.savefig(plot_path, dpi=300)
plt.show()
