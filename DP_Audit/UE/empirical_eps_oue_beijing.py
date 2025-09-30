import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from pathlib import Path

from DP_Audit.compute_empirical_eps import compute_empirical_epsilon_oue
from DP_Audit.UE.ue import ue_mechanism_graph, attack_ue

main_dir = Path(__file__).parent.parent.parent
graph_file = main_dir / "Geolife" / "data" / "beijing_graph.pkl"
result_path = main_dir / "DP_Audit"/ "UE" / "results" / "eps_estimation_oue_Beijing.csv"
attack_result_path = main_dir / "DP_Audit" / "UE" / "results" /"attack_results_oue_Beijing.csv"
ldp_auditor_results = main_dir / "DP_Audit" / "LDP_Auditor" / "results" / "summary_Beijing.csv"
plot_path = main_dir / "DP_Audit" / "UE" / "plots" / "EpsEstimation_OUE_Beijing.png"

### Bisection to approximate bound from Example 5.5
def rad_function(eps, m):
    return (math.exp(eps)-1)/(2*m) * (1 - (math.exp(eps)/(1+math.exp(eps)))**(m-1))

def find_epsilon(rad, m, tol=1e-6):
    eps_min = 0.0
    eps_max = 9.0  # ceiling for eps for OUE
    while eps_max - eps_min > tol:
        eps_mid = (eps_min + eps_max)/2
        if rad_function(eps_mid, m) - rad > 0:
            eps_max = eps_mid
        else:
            eps_min = eps_mid
    return (eps_min + eps_max)/2


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

reros = []
u_reros = []
empirical_epsilons = []
empirical_epsilons_std = []

for epsilon in epsilons:
    results = []

    for j in range(1):  # Repeat several times
        rero = 0
        for node in node_list:
            perturbed_targets = [ue_mechanism_graph(node, node_list, epsilon, True) for _ in range(J)]
            guesses_idx = np.array([attack_ue(pt, m) for pt in perturbed_targets])
            guessed_nodes = node_array[guesses_idx]
            rero += np.sum(guessed_nodes == node) / J
        rero_mean = rero / m
        u_rero_mean = rero_mean - 1/m

        empirical_epsilon = find_epsilon(u_rero_mean, m)
        results.append(empirical_epsilon)

    reros.append(rero_mean)
    u_reros.append(u_rero_mean)
    empirical_epsilons.append(np.mean(results))
    empirical_epsilons_std.append(np.std(results))

    print(f"Eps: {epsilon}, empirical eps: {np.mean(results)}, std: {np.std(results)}")

with open(attack_result_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epsilon", "ReRo", "U-ReRo"])
    for eps, rero, u_rero, in zip(epsilons, reros, u_reros):
        writer.writerow([eps, rero, u_rero])

with open(result_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["eps", "empirical_eps", "std"])
    for eps, emp_eps, std in zip(epsilons, empirical_epsilons, empirical_epsilons_std):
        writer.writerow([eps, emp_eps, std])