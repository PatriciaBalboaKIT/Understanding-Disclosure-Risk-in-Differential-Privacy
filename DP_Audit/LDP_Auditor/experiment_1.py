# general imports
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import logging

# Our imports
from ldp_audit.base_auditor import LDPAuditor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

# Main audit results
def run_main_experiments(nb_trials: int, alpha: float, lst_protocols: list, lst_seed: list, lst_k: list, lst_eps: list, delta: float, analysis: str):
    
    # Initialize dictionary to save results
    results = {
            'seed': [],
            'protocol': [],
            'k': [],
            'delta': [],
            'epsilon': [],    
            'eps_emp': []
            }

    # Initialize LDP-Auditor
    auditor = LDPAuditor(nb_trials=nb_trials, alpha=alpha, epsilon=lst_eps[0], delta=delta, k=lst_k[0], random_state=lst_seed[0], n_jobs=-1)

    # Run the experiments
    for seed in lst_seed:    
        logging.info(f"Running experiments for seed: {seed}")
        for k in lst_k:
            for epsilon in lst_eps:
                
                # Update the auditor parameters
                auditor.set_params(epsilon=epsilon, k=k, random_state=seed)

                for protocol in tqdm(lst_protocols, desc=f'seed={seed}, k={k}, epsilon={epsilon}'):    
                    eps_emp = auditor.run_audit(protocol)                    
                    results['seed'].append(seed)
                    results['protocol'].append(protocol)
                    results['k'].append(k)
                    results['delta'].append(delta)
                    results['epsilon'].append(epsilon)
                    results['eps_emp'].append(eps_emp)

    # Convert and save results in csv file
    df = pd.DataFrame(results)
    df.to_csv('results/ldp_audit_results_{}.csv'.format(analysis), index=False)

    # Group by protocol, k, epsilon and compute mean and std of eps_emp across seeds
    summary = (
    df.groupby(['protocol', 'k', 'epsilon'])['eps_emp']
        .agg(avg_eps_emp='mean', std_eps_emp='std')
        .reset_index()
    )

    # Porto
    summary_porto = summary[summary['k'] == 3052].drop(columns='k')
    summary_porto.to_csv('results/summary_Porto.csv', index=False)

    # Beijing
    summary_beijing = summary[summary['k'] == 5356].drop(columns='k')
    summary_beijing.to_csv('results/summary_Beijing.csv', index=False)
    
    return df

if __name__ == "__main__":
    ## General parameters
    lst_eps = range(1,20)
    lst_k = [3052, 5356]
    lst_seed = range(5)
    nb_trials = int(1e6)
    alpha = 1e-2

    ## pure LDP protocols
    pure_ldp_protocols = ['GRR', 'SS', 'OUE']
    delta = 0.0 
    analysis_pure = 'main_pure_ldp_protocols'
    df_pure = run_main_experiments(nb_trials, alpha, pure_ldp_protocols, lst_seed, lst_k, lst_eps, delta, analysis_pure)
