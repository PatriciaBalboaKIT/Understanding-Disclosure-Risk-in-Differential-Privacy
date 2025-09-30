import haiku as hk
import jax
import jax.numpy as jnp
import optax
from ml_collections import config_dict
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import csv
import hashlib

from absl import logging
from collections import defaultdict

logging.set_verbosity(logging.ERROR)  # or logging.FATAL

from DPSGD_noAux.rdp import get_rdp_epsilon


config = config_dict.ConfigDict()
config.noise_multiplier = 27  # @param, change this for different eps
config.l2_norm_clip = 0.1  # @param
config.epochs = 100  # @param
config.learning_rate = 9.0  # @param.
config.num_in_prior = 8  # @param, Attack base rate will be 1 / config.num_in_prior.
config.batch_size = 1000  # @param
config.q = 1 # @param
config.delta = 1e-5
config.steps = int(config.epochs / config.q)

# Total size of the training dataset. Determined by config.batch_size and
# config.q. For convenience of the attack, which requires some conditions on
# batch sizes, we require config.batch_size to be divisible by config.total_num.
config.total_num = int(config.batch_size / config.q)

# Generate orders used in RDP accounting
orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))

# Get privacy budget for the above configuration.
eps, opt_order = get_rdp_epsilon(
    config.q,
    config.noise_multiplier,
    config.steps,
    config.delta,
    orders,
)
print(f'Epsilon: {eps:.10f}')

attack_config = config_dict.ConfigDict()
attack_config.deduct_fixed_set_grads = True  # @param
attack_config.rescale_by_batch_size = True  # @param


###################### SEED ######################################
def deterministic_seed(correct_idx, run_idx):
    key = f"{correct_idx}-{run_idx}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big")  # 32-bit int



###################### DATA #######################################
def load_dataset(
    split: str,
    *,
    is_training: bool,
    batch_size: int,
    total_num: int,
    start_idx: int = 0,
    repeat: bool = False,
):
  """Loads the MNIST dataset as a generator of batches."""
  ds = tfds.load(
      'mnist', split=split + f'[{start_idx}:{start_idx+total_num}]'
  ).cache()
  ds = ds.batch(batch_size)
  if repeat:
    ds = ds.repeat()
  return iter(tfds.as_numpy(ds))


# Generate the fixed set if config.total_num > 1. Otherwise it doesn't exist and
#  we train only on the sample selected from the prior set.
if config.total_num > 1:
  train_data = load_dataset(
      split='train',
      is_training=True,
      repeat=False,
      batch_size=config.batch_size,
      total_num=config.total_num - 1,
      start_idx=0,
  )

  fixed_images = []
  fixed_labels = []
  for curr_train_batch in train_data:
    fixed_images.extend(curr_train_batch['image'] / 255.0)
    fixed_labels.extend(curr_train_batch['label'])
  fixed_images, fixed_labels = np.array(fixed_images), np.array(fixed_labels)


# Generate prior dataset.
prior_data = load_dataset(
    split='train',
    is_training=True,
    repeat=True,
    batch_size=config.num_in_prior,
    total_num=config.num_in_prior,
    start_idx=config.total_num,
)
prior_batch = next(prior_data)

prior_images = prior_batch['image'] / 255.0
prior_labels = prior_batch['label']

# Load MNIST test set
ds = tfds.load('mnist', split='test', batch_size=-1, as_supervised=True)
test_images, test_labels = tfds.as_numpy(ds)
test_images = test_images.astype(np.float32) / 255.0  # Normalize
test_images = jnp.reshape(test_images, (test_images.shape[0], -1))  # Flatten
test_labels = jnp.array(test_labels)
##############################################################

################## MODEL ####################################
def net_fn(x):
  """Standard MLP network."""
  mlp = hk.Sequential([
      hk.Flatten(),
      hk.Linear(10),
      jax.nn.relu,
      hk.Linear(10),
  ])
  return mlp(x)

net = hk.without_apply_rng(hk.transform(net_fn))
opt = optax.sgd(config.learning_rate)

def broadcast_axis(data, ndims, axis):
  newshape = [1] * ndims
  newshape[axis] = -1
  return data.reshape(*newshape)

@jax.jit
def loss(params, batch):
  """Cross-entropy loss."""
  inputs, targets, unused_is_fixed = batch
  # Inputs scaled to [-1, 1].
  inputs = 2.0 * inputs - 1.0
  logits = net.apply(params, inputs)
  labels = jax.nn.one_hot(targets, 10)
  log_probs = jax.nn.log_softmax(logits)
  softmax_xent_per_example = -jnp.sum(labels * log_probs, axis=1)
  return jnp.mean(softmax_xent_per_example)

@jax.jit
def clipped_grad(params, l2_norm_clip, single_example_batch):
  """Evaluate gradient for a single-example batch and clip its grad norm."""
  # Compute loss and gradient for a single example.
  loss_val, grads = jax.value_and_grad(loss)(params, single_example_batch)
  # Flatten gradient tree and compute the norm.
  nonempty_grads, tree_def = jax.tree_util.tree_flatten(grads)
  total_grad_norm = jnp.linalg.norm(
      jnp.array([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
  )
  divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.0)
  # Normalize gradient to have a maximium norm of l2_norm_clip.
  normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
  return (
      jax.tree_util.tree_unflatten(tree_def, normalized_nonempty_grads),
      loss_val,
  )

#@functools.partial(jax.jit, static_argnums=(3, 4, 5))
@jax.jit
def privatise_gradient(
    params, batch, rng, l2_norm_clip, noise_multiplier, batch_size
):
  """Return differentially private gradients for params, evaluated on batch."""
  # Compute individual sample clipped gradients over a batch.
  clipped_grads, loss_vals = jax.vmap(clipped_grad, (None, None, 0), (0, 0))(
      params, l2_norm_clip, batch
  )
  # Aggregate, add noise, and average these clipped gradients.
  clipped_grads_flat, grads_treedef = jax.tree_util.tree_flatten(clipped_grads)
  aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
  rngs = jax.random.split(rng, len(aggregated_clipped_grads))
  noised_aggregated_clipped_grads = [
      g + l2_norm_clip * noise_multiplier * jax.random.normal(r, g.shape)
      for r, g in zip(rngs, aggregated_clipped_grads)
  ]
  normalized_noised_aggregated_clipped_grads = [
      g / batch_size for g in noised_aggregated_clipped_grads
  ]
  return (
      jax.tree_util.tree_unflatten(
          grads_treedef, normalized_noised_aggregated_clipped_grads
      ),
      loss_vals,
      clipped_grads,
  )

def compute_epsilon(
    steps, num_examples, batch_size, noise_multiplier, target_delta=1e-5
):
  """Compute privacy budget at a given step."""
  q = batch_size / float(num_examples)
  orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
  eps, _ = get_rdp_epsilon(q, noise_multiplier, steps, target_delta, orders)
  return eps

def shape_as_image(batch, dummy_dim=False):
  """Reshape an image in a batch -- useful when we vmap the clipping operation."""
  inputs, targets, is_fixed = batch
  target_shape = (-1, 1, 28, 28, 1) if dummy_dim else (-1, 28, 28, 1)
  return jnp.reshape(inputs, target_shape), targets, is_fixed

#@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7))
@jax.jit
def private_update(
    params,
    opt_state,
    batch,
    rng,
    batch_size,
    l2_norm_clip,
    noise_multiplier
):
  """Update model parameters with a privatized gradient."""

  # Compute the private gradient.
  (private_grad, loss_vals, clipped_grads) = privatise_gradient(
      params, batch, rng, l2_norm_clip, noise_multiplier, batch_size
  )

  # Update model parameters with private gradient.
  updates, opt_state = opt.update(private_grad, opt_state)
  new_params = optax.apply_updates(params, updates)

  # For the attack we will track the clipped gradients over the fixed inputs.
  _, _, is_fixed = batch

  # Sum fixed clipped gradients (known to the attacker).
  summed_fixed_clipped_grads = jax.tree_util.tree_map(
      lambda x: (broadcast_axis(is_fixed, x.ndim, 0) * x).sum(0), clipped_grads
  )
  # Average sum over batch size.
  avg_fixed_clipped_grads = jax.tree_util.tree_map(
      lambda x: x / batch_size, summed_fixed_clipped_grads
  )
  # Attack computes maybe_noisy_target_grad which is either:
  # * random noise if the target is not in batch.
  # * (target_gradient / batch_size) + noise if the target is in the batch.
  maybe_noisy_target_grad = jax.tree_util.tree_map(
      lambda x, y: x - y, private_grad, avg_fixed_clipped_grads
  )

  return (
      new_params,
      opt_state,
      loss_vals.mean(),
      private_grad,
      maybe_noisy_target_grad,
  )

def train_fn(attack_config, config):
  """Train a differentially private model."""

  # Grab the target image from the prior set.
  target_image = prior_images[attack_config.target_idx_from_prior][None, ...]
  target_label = prior_labels[attack_config.target_idx_from_prior][None, ...]

  # Create a training set of images. We also include a marker identifying
  # if an input belongs to the fixed set.
  if config.total_num > 1:
    train_images = np.concatenate((fixed_images, target_image))
    train_labels = np.concatenate((fixed_labels, target_label))

    is_fixed = np.concatenate(
        (np.ones((len(fixed_images),)), np.zeros((len(target_image),)))
    )
    total_num = config.total_num
    batch_size = config.batch_size

  else:
    train_images = target_image
    train_labels = target_label
    total_num = 1
    batch_size = 1
    is_fixed = np.zeros((len(target_image),))

  ds = tf.data.Dataset.from_tensor_slices(
      (train_images, train_labels, is_fixed)
  )
  ds = ds.shuffle(1000).batch(batch_size, drop_remainder=True)
  ds = ds.repeat()
  ds = iter(tfds.as_numpy(ds))

  # Set up optimiser.
  opt = optax.sgd(config.learning_rate)

  # Initialise a set of model parameters.
  rng = jax.random.PRNGKey(config.seed)
  params = net.init(rng, np.ones((1, 28, 28, 1)))
  opt_state = opt.init(params)

  # We train a differentially private model and keep track of information
  # available to the attacker:
  # * model parameters.
  # * private gradient.
  # * private gradient with clipped gradients from fixed set subtracted.
  info_for_attacker = []
  for step in range(1, config.steps + 1):
    batch = next(ds)
    _, _, is_fixed = batch

    rng, _ = jax.random.split(rng)

    # Compute updates.
    (opt_params, opt_state, loss_val, private_grad, maybe_noisy_target_grad) = (
        private_update(
            params,
            opt_state,
            shape_as_image(batch, dummy_dim=True),
            rng,
            batch_size,
            config.l2_norm_clip,
            config.noise_multiplier
        )
    )

    # Determine privacy loss so far.
    eps = compute_epsilon(
        step, total_num, batch_size, config.noise_multiplier, config.delta
    )

    # Track information available to the attacker.
    info_for_attacker.append((params, private_grad, maybe_noisy_target_grad))

    # Update model parameters.
    params = opt_params

  # Determine privacy loss.
  eps = compute_epsilon(
      step, total_num, batch_size, config.noise_multiplier, config.delta
  )
  # print(eps)

  # Run inference
  test_images_scaled = 2.0 * test_images - 1.0
  logits = net.apply(params, test_images_scaled)
  preds = jnp.argmax(logits, axis=-1)

  # Compute accuracy
  accuracy = jnp.mean(preds == test_labels)
  print("Test accuracy:", float(accuracy))

  return info_for_attacker, eps, accuracy

def reconstruction_upper_bound(pmode, q, noise_mul, steps, mc_samples=10000):
    x = np.random.normal(0.0, noise_mul, (mc_samples,steps))
    per_step_log_ratio= np.log(1-q + q*(np.exp((-(x-1.0)**2 + (x)**2)/(2*noise_mul**2))))
    log_ratio=np.sum(per_step_log_ratio,axis=1)
    log_ratio=np.sort(log_ratio)
    r=np.exp(log_ratio)
    upper_bound=max(0.0,1-(1-pmode)*np.mean(r[:int(mc_samples*(1-pmode))]))
    return min(1.0, upper_bound)

@jax.jit
def compute_dot_prod(g1, g2):
  """Compute dot product between two trees."""
  return jnp.array(
      jax.tree_util.tree_leaves(
          jax.tree_util.tree_map(lambda x, y: jnp.sum(x * y), g1, g2)
      )
  ).sum()
###################################################################

######################## ATTACK ###################################
def reconstruction_attack(config, attack_config, target_idx, run_idx, aux_set):
  """Train a differentially private model and perform a reconstruction attack."""
  # Train a (private) model.
  info_for_attacker, eps, accuracy = train_fn(attack_config, config)

  # Extract information available to the attacker.
  (
      params_over_time,
      private_grad_over_time,
      maybe_noisy_target_grad_over_time,
  ) = zip(*info_for_attacker)

  # Loop over all candidate images in prior.
  dot_prod_cands = []
  improved_dot_prod_cands = []
  for i, (xp, yp) in enumerate(zip(prior_images, prior_labels)):
    # Init value we will use to check decide which image in the prior was used.
    dot_prod_sum = 0
    dot_prod_agg = []

    # Loop over all update steps.
    for params, private_grad, maybe_noisy_target_grad in zip(
        params_over_time,
        private_grad_over_time,
        maybe_noisy_target_grad_over_time,
    ):
      # Compute clipped gradient of candidate image.
      candidate_clipped_grad, _ = clipped_grad(
          params, config.l2_norm_clip, (xp[None, ...], yp[None, ...], None)
      )

      # If adversary knows the batch size, we can divide the candidate
      # gradient by the batch size.
      if attack_config.rescale_by_batch_size:
        candidate_clipped_grad = jax.tree_util.tree_map(
            lambda x: x / config.batch_size, candidate_clipped_grad
        )

      # If the attacker knows which other examples were used in training, then
      # we can use maybe_noisy_target_grad otherwise use private_grad
      if attack_config.deduct_fixed_set_grads:
        dot_prod_val = compute_dot_prod(
            maybe_noisy_target_grad, candidate_clipped_grad
        )
      else:
        dot_prod_val = compute_dot_prod(private_grad, candidate_clipped_grad)

      dot_prod_sum += dot_prod_val
      dot_prod_agg.append(dot_prod_val)

    # Append the dot product sum.
    dot_prod_cands.append(dot_prod_sum)

    # Improved attack -- Split into batches representing epoch and take the
    # max value from each. Note there is a mismatch between theory and practice
    # here since DP-SGD accounting assumes data is sub-sampled not shuffled.
    filtered_dot_prod_sum = sum(np.max(np.array(np.split(np.array(dot_prod_agg), config.epochs)), axis=-1))
    improved_dot_prod_cands.append(filtered_dot_prod_sum)

  # Take the argmax from that aux set
  scores_aux_set = [improved_dot_prod_cands[idx] for idx in aux_set]
  max_idx = np.argmax(scores_aux_set)
  max_aux_set_idx = aux_set[max_idx]
  assert max_aux_set_idx in aux_set
  
  if improved_dot_prod_cands[max_aux_set_idx] > np.mean(improved_dot_prod_cands):
    print("Chance at returning the right one")
    return max_aux_set_idx, info_for_attacker, accuracy
  else:
    print("Defo wrong")
    other_indexes = [idx for idx in range(config.num_in_prior) if idx not in aux_set]
    return np.random.choice(other_indexes), info_for_attacker, accuracy
  

def reco_attack_u_rero(info_for_attacker, target_idx, aux_set):
  (
      params_over_time,
      private_grad_over_time,
      maybe_noisy_target_grad_over_time,
  ) = zip(*info_for_attacker)

  # Loop over all candidate images in prior.
  dot_prod_cands = []
  improved_dot_prod_cands = []
  for i, (xp, yp) in enumerate(zip(prior_images, prior_labels)):
    # Init value we will use to check decide which image in the prior was used.
    dot_prod_sum = 0
    dot_prod_agg = []

    # Loop over all update steps.
    for params, private_grad, maybe_noisy_target_grad in zip(
        params_over_time,
        private_grad_over_time,
        maybe_noisy_target_grad_over_time,
    ):
      # Compute clipped gradient of candidate image.
      candidate_clipped_grad, _ = clipped_grad(
          params, config.l2_norm_clip, (xp[None, ...], yp[None, ...], None)
      )

      # If adversary knows the batch size, we can divide the candidate
      # gradient by the batch size.
      if attack_config.rescale_by_batch_size:
        candidate_clipped_grad = jax.tree_util.tree_map(
            lambda x: x / config.batch_size, candidate_clipped_grad
        )

      # If the attacker knows which other examples were used in training, then
      # we can use maybe_noisy_target_grad otherwise use private_grad
      if attack_config.deduct_fixed_set_grads:
        dot_prod_val = compute_dot_prod(
            maybe_noisy_target_grad, candidate_clipped_grad
        )
      else:
        dot_prod_val = compute_dot_prod(private_grad, candidate_clipped_grad)

      dot_prod_sum += dot_prod_val
      dot_prod_agg.append(dot_prod_val)

    # Append the dot product sum.
    dot_prod_cands.append(dot_prod_sum)

    # Improved attack -- Split into batches representing epoch and take the
    # max value from each. Note there is a mismatch between theory and practice
    # here since DP-SGD accounting assumes data is sub-sampled not shuffled.
    filtered_dot_prod_sum = sum(np.max(np.array(np.split(np.array(dot_prod_agg), config.epochs)), axis=-1))
    improved_dot_prod_cands.append(filtered_dot_prod_sum)

  # Take the argmax from that aux set
  scores_aux_set = [improved_dot_prod_cands[idx] for idx in aux_set]
  max_idx = np.argmax(scores_aux_set)
  max_aux_set_idx = aux_set[max_idx]
  assert max_aux_set_idx in aux_set
  
  if improved_dot_prod_cands[max_aux_set_idx] > np.mean(improved_dot_prod_cands):
    print("Chance at returning the right one")
    return max_aux_set_idx
  else:
    print("Defo wrong")
    other_indexes = [idx for idx in range(config.num_in_prior) if idx not in aux_set]
    return np.random.choice(other_indexes)
  


############################ AUX ##################################
# The adversary knows what number is displayed on the target image
aux_dict = defaultdict(list)
for idx, label in enumerate(prior_labels):
  aux_dict[label].append(idx)
print(aux_dict)

############################ EXECUTION ################################
runs = 1000

# Index of prior point we include from the prior set.
correct_idx = int(input("Correct idx from 0 to 7"))
attack_config.target_idx_from_prior = correct_idx

print(f"\n\n Correct idx: {correct_idx}\n\n")

aux_label = prior_labels[correct_idx]
aux_set = aux_dict[aux_label]

# Run attack and return if we correctly identified the target image from
# the prior.
round_reros = []
round_correction = []
round_acc = []
for run_idx in range(runs):
  config.seed = deterministic_seed(correct_idx, run_idx)
  improved_attack_cand_idx, info_for_attacker, accuracy = reconstruction_attack(config, attack_config, correct_idx, run_idx, aux_set)
  round_reros.append(improved_attack_cand_idx == correct_idx)
  round_acc.append(accuracy)

  # U-ReRo
  model_results = []
  for target_idx in range(0, config.num_in_prior):
    predicted_index = reco_attack_u_rero(info_for_attacker, target_idx, aux_set)
    model_results.append(predicted_index == target_idx)
  round_correction.append(np.mean(model_results))

rero = np.mean(round_reros)
correction_term = np.mean(round_correction)
u_rero = rero - correction_term
test_accuracy = np.mean(round_acc)

rub = reconstruction_upper_bound(1/config.num_in_prior, config.q, config.noise_multiplier, config.steps)
balle_bound = np.exp(eps) * 1/config.num_in_prior
print(f"\nHayes bound: {rub:.3f}, ReRo: {rero:.3f}, U-ReRo: {u_rero:.3f},  Balle bound: {balle_bound:.3f}, Test accuracy: {test_accuracy}")

with open(f'./DPSGD_partAux/results/mnist_raw/mnist_eps{eps}_idx{correct_idx}_partAux_big.csv', mode='w', newline='') as file:
  writer = csv.writer(file)
  writer.writerow(["Eps", 'ReRo', 'U-ReRo','Hayes bound', 'Balle bound'])
  writer.writerow([eps, rero, u_rero, rub, balle_bound]) 