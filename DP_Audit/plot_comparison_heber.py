import pandas as pd
import matplotlib.pyplot as plt

# === Load your results ===
df_ours = pd.read_csv("./SS/results/eps_estimation_Beijing.csv")
epsilons_ours = df_ours["eps"][:19]
empirical_eps_ours = df_ours["empirical_eps"][:19]
std_ours = df_ours["std"][:19]

# === Load competitor results ===
df_comp = pd.read_csv("./LDP_Auditor/results/summary_Beijing.csv")

# === Choose protocol ===
df_comp_filtered = df_comp[(df_comp["protocol"] == "SS")]
df_comp_filtered = df_comp_filtered.sort_values(by="epsilon")
epsilons_comp = df_comp_filtered["epsilon"][:19]
empirical_eps_comp = df_comp_filtered["avg_eps_emp"][:19]
std_comp = df_comp_filtered["std_eps_emp"][:19]

# === Plot both ===
plt.figure(figsize=(8, 6))
plt.errorbar(
    epsilons_ours,
    empirical_eps_ours,
    yerr=std_ours,
    fmt='x',
    color='red',
    capsize=5,
    label=r"Ours"
)

plt.errorbar(
    epsilons_comp,
    empirical_eps_comp,
    yerr=std_comp,
    fmt='o',
    color='blue',
    capsize=5,
    label=f"LDP Auditor (Arcolezi et al.)"
)
plt.plot(epsilons_ours, epsilons_ours, color='gray', linestyle="--", label=r"Real $\varepsilon$")
plt.xlabel(r"Theoretical $\varepsilon$", fontsize=18)
plt.ylabel(r"Empirical $\varepsilon$", fontsize=18)
plt.legend(fontsize=14)
plt.grid(True)
plt.xlim(0,20)
plt.ylim(0,20)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.show()
