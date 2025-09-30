import numpy as np
from numba import jit

def ss_mechanism_graph(true_node, all_nodes, epsilon):
    k = len(all_nodes)
    index = all_nodes.index(true_node)
    return SS_Client(index, k, epsilon)

@jit(nopython=True)
def SS_Client(input_data, k, epsilon):
    """
    Subset Selection (SS) protocol [1,2].

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :return: set of sub_k sanitized values.
    """

    # Validations
    if input_data < 0 or input_data >= k:
        raise ValueError('input_data (integer) should be in the range [0, k-1].')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if epsilon > 0:

        # Mapping domain size k to the range [0, ..., k-1]
        domain = np.arange(k)

        # SS parameters
        sub_k = int(max(1, np.rint(k / (np.exp(epsilon) + 1))))
        p_v = sub_k * np.exp(epsilon) / (sub_k * np.exp(epsilon) + k - sub_k)

        # SS perturbation function
        rnd = np.random.random()
        sub_set = np.zeros(sub_k, dtype='int64')
        if rnd <= p_v:
            sub_set[0] = int(input_data)
            sub_set[1:] = np.random.choice(domain[domain != input_data], size=sub_k-1, replace=False)
            return sub_set

        else:
            return np.random.choice(domain[domain != input_data], size=sub_k, replace=False)

    else:
        raise ValueError('epsilon (float) needs a numerical value greater than 0.')

@jit(nopython=True)
def attack_ss(ss):
    """
    Privacy attack to Subset Selection (SS) protocol.

    Parameters:
    ----------
    ss : array
        Obfuscated subset of values.

    Returns:
    -------
    int
        A random inference of the true value.
    """
                
    return np.random.choice(ss)