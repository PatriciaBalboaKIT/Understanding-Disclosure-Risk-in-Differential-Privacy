import math

### Newest bound: uniform prior + eta=0 (Corollary 5.3) ########
def compute_empirical_epsilon_unif(u_rero, m):
    eps = float('nan')
    try:
        eps = math.log((u_rero * m + 1) / (1 - u_rero * (m /(m-1)))) 
    except:
        pass

    return eps if eps >= 0 else 0

import math

def compute_empirical_epsilon_oue(u_rero, m):
    eps_values = []

    # First term: Corollary 5.3
    try:
        term1 = math.log((u_rero * m + 1) / (1 - u_rero * (m /(m-1)))) 
        eps_values.append(term1)
    except:
        pass

    # Second term: OUE special bound
    try:
        g_pi = (m - 1) / m
        term2 = math.log((2 * u_rero * g_pi + 1) / (1 - 2*u_rero*g_pi))
        eps_values.append(term2)
    except:
        pass

    if eps_values:
        # returns either the max or the only one that worked
        eps = max(eps_values)
        return eps if eps >= 0 else 0
    else:
        return float('nan')
