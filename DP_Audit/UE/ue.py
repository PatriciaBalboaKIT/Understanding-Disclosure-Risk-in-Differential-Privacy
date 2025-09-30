import numpy as np
import random
from numba import jit, prange

def ue_mechanism_graph(true_node, all_nodes, epsilon, optimal=True):
    """
    Graph wrapper for UE Client.

    Parameters:
    ----------
    true_node : hashable
        The true graph node (e.g., a node ID).
    all_nodes : list
        List of all possible graph nodes.
    epsilon : float
        Privacy budget.
    optimal : bool
        Whether to use Optimized Unary Encoding (OUE).

    Returns:
    -------
    sanitized_vec : np.ndarray
        Unary encoded and perturbed vector.
    """
    k = len(all_nodes)
    index = all_nodes.index(true_node)
    return UE_Client(index, k, epsilon, optimal)


		
#@jit(nopython=True)
def UE_Client(input_data, k, epsilon, optimal=True):
    """
    Unary Encoding (UE) protocol, a.k.a. Basic One-Time RAPPOR (if optimal=False) [1]

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :return: sanitized UE vector.
    """

    # Validations
    if input_data != None:
        if input_data < 0 or input_data >= k:
            raise ValueError('input_data (integer) should be in the range [0, k-1].')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if epsilon > 0:

        # Symmetric parameters (p+q = 1)
        p = np.exp(epsilon/2) / (np.exp(epsilon/2) + 1)
        q = 1 - p

        # Optimized parameters
        if optimal:
            p = 1 / 2
            q = 1 / (np.exp(epsilon) + 1)

        # Unary encoding
        input_ue_data = np.zeros(k)
        if input_data != None:
            input_ue_data[input_data] = 1

        # UE perturbation function
        oh_vec = np.random.choice([1, 0], size=k, p=[q, 1-q])  # If entry is 0, flip with prob q
        oh_vec[input_data] = 0                  # BROKEN implementation does not have this
        if random.random() < p:
            oh_vec[input_data] = 1

        '''    
        # Initializing a zero-vector
        sanitized_vec = np.zeros(k)
        for ind in range(k):
            if input_ue_data[ind] != 1:
                rnd = np.random.random()
                if rnd <= q:
                    sanitized_vec[ind] = 1
            else:
                rnd = np.random.random()
                if rnd <= p:
                    sanitized_vec[ind] = 1
        return sanitized_vec'''
        return oh_vec

    else:
        raise ValueError('epsilon (float) needs a numerical value greater than 0.')


@jit(nopython=True)
def attack_ue(ue_val, k):
    """
    Privacy attack to Unary Encoding (UE) protocols.

    Parameters:
    ----------
    ue_val : array
        Obfuscated vector.
    k : int
        Domain size.

    Returns:
    -------
    int
        A random inference of the true value.
    """

    if np.sum(ue_val) == 0:
        return np.random.randint(k)
    else:
        return np.random.choice(np.where(ue_val == 1)[0])