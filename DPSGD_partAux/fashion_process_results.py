import pandas as pd
from pathlib import Path

main_dir = Path(__file__).parent.parent
raw_results_folder = main_dir / "DPSGD_partAux" / "results" / "fashion_raw"

# Combine results of all images in candidate set
candidate_set_size = 8
epsilons = [0.512063674748734, 1.0125506277526433, 1.5567628736984664, 2.051797484856534, 5.0159472915120995, 10.725509682302643]

for eps in epsilons:
    output_file = main_dir / "DPSGD_partAux" / "results" / f"fashion_eps{eps}.csv"
    dfs = []
    for i in range(candidate_set_size):
        df = pd.read_csv(raw_results_folder / f"fashion_eps{eps}_idx{i}_partAux_big.csv")
        df = df.apply(pd.to_numeric, errors="coerce")
        dfs.append(df)
    combined = pd.concat(dfs)
    averages = combined.mean(axis=0)
    averages.to_frame().T.to_csv(output_file, index=False)