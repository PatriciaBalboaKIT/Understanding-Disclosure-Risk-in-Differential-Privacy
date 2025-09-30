import json
import os
import pickle

from datetime import datetime
from pathlib import Path

from Geolife.constants import BEIJING_CENTERPOINT, BEIJING_RADIUS
from Porto.roadgraph import roadgraph


# file imports
current_dir = Path(__file__).parent.parent
geolife_constants_file = current_dir / "Geolife" / "constants.py"
geolife_directory = current_dir / "Geolife" / "Data"
processed_geolife_data = current_dir / "Geolife" / "data" / "geolife_trajectories.json"
graph_pkl = current_dir / "Geolife" / "data" / "beijing_graph.pkl"


### FUNCIONES ###

# Función para leer una trayectoria desde un archivo Geolife .plt
def read_trajectory(file_path):
    trajectory = []

    # Leer las líneas del archivo, omitiendo las primeras 6 (metadata)
    with open(file_path, 'r') as f:
        lines = f.readlines()[6:]
        for line in lines:
            parts = line.strip().split(',')
            lat = float(parts[0])
            lon = float(parts[1])
            time_str = parts[5] + ' ' + parts[6]  # Fecha y hora
            time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            unix_timestamp = int(time.timestamp())
            trajectory.append((lat, lon, unix_timestamp))

    return trajectory

# Función para obtener todas las trayectorias desde archivos .plt
def get_all_trajectories(directory):
    all_trajectories = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".plt"):
                trajectory = read_trajectory(os.path.join(root, file))
                if len(trajectory) > 0:  # Asegúrate de que la trayectoria no esté vacía
                    all_trajectories.append(trajectory)
    return all_trajectories

def process_trajectory(trajectory):
    times = []
    latitudes = []
    longitudes = []

    # Extraer coordenadas y tiempos
    for point in trajectory:
        lat, lon, time = point
        times.append(time)
        latitudes.append(lat)
        longitudes.append(lon)

    start_time = times[0]

    # Crear lista de coordenadas
    coordinates = [
        [lat, lon] 
        for lat, lon in zip(latitudes, longitudes)
    ]

    return coordinates, start_time

# Guardar las trayectorias en un archivo JSON con el timestamp inicial
def save_trajectories_to_json(trajectories, output_file):
    data = {}

    for i, traj in enumerate(trajectories):
        traj_name = f"traj{i+1}"
        processed_traj, start_time = process_trajectory(traj)
        data[traj_name] = {
            "coordinates": processed_traj,
            "start_time": start_time
        }

    # Guardar en el archivo JSON
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


def load_trajectories_from_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    # extract all trajectories from Geolife archive
    all_trajectories = get_all_trajectories(geolife_directory)

    # save processed trajectories to JSON
    save_trajectories_to_json(all_trajectories, processed_geolife_data)

    trajectories = load_trajectories_from_json(processed_geolife_data)

    # create roadgraph
    (_, G, _) = roadgraph(BEIJING_CENTERPOINT, BEIJING_RADIUS)
    node_list = list(G.nodes())

    with open(graph_pkl, 'wb') as f:
        pickle.dump(G, f)

    with open(geolife_constants_file, 'a') as f:
        f.write(f'\nM = {len(node_list)}\n')

