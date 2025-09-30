import numpy as np
import random

def grr_mechanism(true_node, all_nodes, p):
    coin = random.random()

    if coin <= p:
        return true_node
    else:
        node_array = np.array(all_nodes)
        filtered = node_array[node_array != true_node]
        return np.random.choice(filtered)