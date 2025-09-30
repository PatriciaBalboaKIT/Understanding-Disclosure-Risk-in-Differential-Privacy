import csv
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

from pathlib import Path

from DP_Audit.compute_empirical_eps import compute_empirical_epsilon_unif
from DP_Audit.SS.ss import ss_mechanism_graph, attack_ss

main_dir = Path(__file__).parent.parent.parent
graph_file = main_dir / "Geolife" / "data" / "beijing_graph.pkl"
result_path = main_dir / "DP_Audit"/ "SS" / "results" / "eps_estimation_Beijing.csv"
attack_result_path = main_dir / "DP_Audit" / "SS" / "results" /"attack_results_Beijing.csv"
ldp_auditor_results = main_dir / "DP_Audit" / "LDP_Auditor" / "results" / "summary_Beijing.csv"
plot_path = main_dir / "DP_Audit" / "SS" / "plots" / "EpsEstimation_Beijing.png"

### Bisection to approximate bound from Example 5.5
def rad_function(eps, m):
    w = max(1, np.floor(m / (np.exp(eps) + 1)))
    p = (w * np.exp(eps))/(w*np.exp(eps) + m - w)
    value = (p * m - w) / (m * w)
    return value

def find_epsilon(rad, m, tol=1e-6):
    eps_min = 0.0
    eps_max = 20.0  # ceiling for eps
    while eps_max - eps_min > tol:
        eps_mid = (eps_min + eps_max)/2
        if rad_function(eps_mid, m) - rad > 0:
            eps_max = eps_mid
        else:
            eps_min = eps_mid
    return (eps_min + eps_max)/2


###### Compute empirical epsilon #######
epsilons = range(1,20)
empirical_epsilons = []
empirical_epsilons_std = []

MC_samples = 1000000

with open(graph_file, 'rb') as f:
    G = pickle.load(f)
node_list = list(G.nodes())
node_array = np.array(node_list)
m = len(node_list)

J = int(MC_samples / m)

empirical_epsilons = []
empirical_epsilons_std = []

reros = []
u_reros = []

for epsilon in epsilons:
    results = []
    for _ in range(5):  # Repeat several times
        rero = 0
        # I: all samples
        for node in node_list:
            perturbed_targets = [ss_mechanism_graph(node, node_list, epsilon) for _ in range(J)]
            guesses_idx = np.array([attack_ss(pt) for pt in perturbed_targets])
            guessed_nodes = node_array[guesses_idx]
            rero += np.sum(guessed_nodes == node) / J
        rero_mean = rero / m
        u_rero_mean = rero_mean - 1/m

        empirical_epsilon = find_epsilon(u_rero_mean, m)
        results.append(empirical_epsilon)
    
    empirical_epsilons.append(np.mean(results))
    empirical_epsilons_std.append(np.std(results))

    reros.append(rero_mean)
    u_reros.append(u_rero_mean)

    print(f"Eps: {epsilon}, empirical eps: {np.mean(results)}, std: {np.std(results)}")

with open(result_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["eps", "empirical_eps", "std"])
    for eps, emp_eps, std in zip(epsilons, empirical_epsilons, empirical_epsilons_std):
        writer.writerow([eps, emp_eps, std])

with open(attack_result_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epsilon", "ReRo", "U-ReRo"])
    for eps, rero, u_rero, in zip(epsilons, reros, u_reros):
        writer.writerow([eps, rero, u_rero])