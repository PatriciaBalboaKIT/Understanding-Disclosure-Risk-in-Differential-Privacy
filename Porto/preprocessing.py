# Read in the Porto data from train.csv, drop irrelevant columns and save it to porto.pkl for further use.
import pandas as pd
import pickle

from pathlib import Path
from Porto.constants import PORTO_CENTERPOINT, PORTO_RADIUS
from Porto.roadgraph import roadgraph

## file imports
main_dir = Path(__file__).parent.parent
raw_porto_data = main_dir / "Porto" / "Data" / "train.csv"
processed_porto_data = main_dir / "Porto" / "data" / "porto.pkl"
graph_file = main_dir / "Porto" / "data" / "porto_graph.pkl"
porto_constants_file = main_dir / "Porto" / "constants.py"

##### Process raw data
def create_porto_df(raw_data_path=raw_porto_data, output_path=processed_porto_data):
    # read file
    porto_df = pd.read_csv(raw_data_path)

    # remove useless columns
    porto_df = porto_df.drop(["CALL_TYPE", "ORIGIN_CALL", "ORIGIN_STAND", "DAY_TYPE", "MISSING_DATA"], axis=1)

    # save file with Porto trajectories
    porto_df.to_pickle(output_path)


if __name__ == "__main__":
    # Read in the raw data
    create_porto_df()

    # Create and store roadgraph
    (_, G, _) = roadgraph(PORTO_CENTERPOINT, PORTO_RADIUS)
    node_list = list(G.nodes())

    with open(graph_file, 'wb') as f:
        pickle.dump(G, f)
    
    with open(porto_constants_file, 'a') as f:
        f.write(f'\nM = {len(node_list)}\n')
