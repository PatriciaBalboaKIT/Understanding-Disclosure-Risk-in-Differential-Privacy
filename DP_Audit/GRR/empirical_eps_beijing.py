import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

from pathlib import Path

from DP_Audit.compute_empirical_eps import compute_empirical_epsilon_unif
from DP_Audit.GRR.grr import grr_mechanism

# file imports
main_dir = Path(__file__).parent.parent.parent
graph_file = main_dir / "Geolife" / "data" / "beijing_graph.pkl"
result_path = main_dir / "DP_Audit"/ "GRR" / "results" / "eps_estimation_Beijing.csv"
attack_result_path = main_dir / "DP_Audit" / "GRR" / "results" /"attack_results_Beijing.csv"
ldp_auditor_results = main_dir / "DP_Audit" / "LDP_Auditor" / "results" / "summary_Beijing.csv"
plot_path = main_dir / "DP_Audit" / "GRR" / "plots" / f"EpsEstimation_Beijing.png"

###### Compute empirical epsilon #######
epsilons = range(1,20)
MC_samples = 1000000

with open(graph_file, 'rb') as f:
    G = pickle.load(f)
node_list = list(G.nodes())
m = len(node_list)

J = int(MC_samples / m)

empirical_epsilons = []
empirical_epsilons_std = []

reros = []
u_reros = []

for epsilon in epsilons:
    e_exp = math.exp(epsilon)
    p = e_exp / (e_exp + len(node_list) - 1)

    results = []
    for _ in range(5):
        rero = 0
        # I: all samples
        for node in node_list:
            # J: 1,000,000 / all_samples
            predictions = np.array([grr_mechanism(true_node=node, all_nodes=node_list, p=p) for _ in range(J)])
            rero += np.sum(predictions == node) / J
        rero_mean = rero / m
        u_rero_mean = rero_mean - 1/m

        # Estimate empirical epsilon
        empirical_epsilon = compute_empirical_epsilon_unif(u_rero=u_rero_mean, m=len(node_list))
        results.append(empirical_epsilon)

    reros.append(rero_mean)
    u_reros.append(u_rero_mean)
    
    empirical_epsilons.append(np.mean(results))
    empirical_epsilons_std.append(np.std(results))

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