import jax.numpy as jnp
import numpy as np

from dp_accounting import dp_event
from dp_accounting.rdp import rdp_privacy_accountant

RdpAccountant = rdp_privacy_accountant.RdpAccountant
ORDERS = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))

def compute_epsilons_and_sigmas(q, num_steps, delta, sigma_start=1e-6, sigma_end=10, sigma_step=0.01):
    def get_rdp_epsilon(sampling_probability, noise_multiplier, steps, delta):
        event = dp_event.PoissonSampledDpEvent(
            sampling_probability,
            event=dp_event.GaussianDpEvent(noise_multiplier)
        )
        accountant = RdpAccountant(orders=ORDERS)
        accountant.compose(event, steps)
        epsilon, _ = accountant.get_epsilon_and_optimal_order(delta)
        return epsilon

    sigmas = np.arange(sigma_start, sigma_end, sigma_step)
    epsilons = np.array([get_rdp_epsilon(q, sigma, num_steps, delta) for sigma in sigmas])
    mask = (epsilons >= 0.01) & (epsilons <= 22)
    return sigmas[mask], epsilons[mask]


def generate_pi_distributions(m):
    """
    Generate different attribute distributions pi for a given number of attribute values m.
    
    Returns a dictionary with keys:
        - 'uniform': perfectly uniform
        - 'slightly_nonuniform': slightly perturbed uniform
        - 'middle': some values higher, some lower
        - 'far_from_uniform': one dominant, others small
        - 'very_biased': one very high, others tiny
    """
    # Uniform
    uniform = np.ones(m) / m

    # Slightly non-uniform (small random perturbation)
    perturb = np.random.uniform(-0.05, 0.05, m)
    slightly_nonuniform = uniform + perturb
    slightly_nonuniform = np.clip(slightly_nonuniform, 0, None)
    slightly_nonuniform /= slightly_nonuniform.sum()

    # Middle: some moderately higher, some lower
    middle = np.linspace(0.1, 1.0, m)
    middle /= middle.sum()

    # Far from uniform: one dominant
    far = np.ones(m) * 0.05
    far[0] = 1 - far.sum() + 0.05
    far /= far.sum()

    # Very biased: one very high, rest tiny
    very_biased = np.ones(m) * 1e-3
    very_biased[0] = 1 - very_biased.sum() + 1e-3
    very_biased /= very_biased.sum()

    return {
        'uniform': uniform,
        'slightly_nonuniform': slightly_nonuniform,
        'middle': middle,
        'far_from_uniform': far,
        'very_biased': very_biased
    }