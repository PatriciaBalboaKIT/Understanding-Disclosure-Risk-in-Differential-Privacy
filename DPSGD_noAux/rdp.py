from dp_accounting import dp_event
from dp_accounting.rdp import rdp_privacy_accountant



RdpAccountant = rdp_privacy_accountant.RdpAccountant


def get_rdp_epsilon(
    sampling_probability, noise_multiplier, steps, delta, orders
):
  """Get privacy budget from Renyi DP."""
  event = dp_event.PoissonSampledDpEvent(
      sampling_probability, event=dp_event.GaussianDpEvent(noise_multiplier)
  )
  rdp_accountant = RdpAccountant(orders=orders)
  rdp_accountant.compose(event, steps)
  rdp_epsilon, opt_order = rdp_accountant.get_epsilon_and_optimal_order(delta)
  return rdp_epsilon, opt_order