# Copyright 2021, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An implementation of SCAFFOLD algorithm.
"""
import collections
from typing import Callable, Collection, Optional
import attr

import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy as tfp

from federated_research.dp_ftrl import optimizer_utils
from federated_research.utils import tensor_utils


DEFAULT_SERVER_OPTIMIZER_FN = lambda w: optimizer_utils.SGDServerOptimizer(  # pylint: disable=g-long-lambda
    learning_rate=1.0)
DEFAULT_CLIENT_OPTIMIZER_FN = lambda: tf.keras.optimizers.SGD(learning_rate=0.1)


def _dataset_reduce_fn(reduce_fn, dataset, initial_state_fn):
  return dataset.reduce(initial_state=initial_state_fn(), reduce_func=reduce_fn)


def _for_iter_dataset_fn(reduce_fn, dataset, initial_state_fn):
  """Performs dataset reduce for simulation performance."""
  update_state = initial_state_fn()
  for batch in iter(dataset):
    update_state = reduce_fn(update_state, batch)
  return update_state

def _build_dataset_reduce_fn(simulation_flag=True):
  """Returns a reduce loop function on input dataset."""
  if simulation_flag:
    return _for_iter_dataset_fn
  else:
    return _dataset_reduce_fn

def _build_grad_reduce_fn(simulation_flag=True):
  """Returns a reduce loop function on input dataset."""
  if simulation_flag:
    return _for_iter_dataset_fn
  else:
    return _dataset_reduce_fn


def _unpack_data_label(batch):
  if isinstance(batch, collections.abc.Mapping):
    return batch['x'], batch['y']
  elif isinstance(batch, (tuple, list)):
    if len(batch) < 2:
      raise ValueError('Expecting both data and label from a batch.')
    return batch[0], batch[1]
  else:
    raise ValueError('Unrecognized batch data.')


def _get_model_weights(model):
  if hasattr(model, 'weights'):
    return model.weights
  else:
    return tff.learning.ModelWeights.from_model(model)


@attr.s(eq=False, frozen=True, slots=True)
class BroadcastMessage(object):
  """Structure for tensors broadcasted by server during federated optimization.

  Fields:
  -   `model_weights`: A dictionary of model's trainable tensors.
  -   `dp_clip_norm`: Clip norm for client model delta.
  -   `mean_control_weights`: Average of control variates used for gradient
        correction
  """
  model_weights = attr.ib()
  dp_clip_norm = attr.ib()
  mean_control_weights = attr.ib()


@tf.function
def build_server_broadcast_message(server_state):
  """Builds `BroadcastMessage` for broadcasting.

  Args:
    server_state: A `ServerState`.

  Returns:
    A `BroadcastMessage`.
  """
  return BroadcastMessage(
      model_weights=server_state.model, dp_clip_norm=server_state.dp_clip_norm,
      mean_control_weights=server_state.mean_control_weights)

@attr.s(eq=False, frozen=True, slots=True)
class ClientState(object):
  """Structure for state on the client.

  Attributes:
    client_id: The client id to map the client state back to
      the database hosting client states in the driver file.
    control_weights: The control variate weights for this client.
  """
  client_id = attr.ib()
  control_weights = attr.ib()

@attr.s(eq=False, frozen=True, slots=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `weights_delta`: A dictionary of updates to the model's trainable
      variables.
  -   `control_weights_delta`: Difference in local and updated controls.
  -   `client_weight`: Weight to be used in a weighted mean when
      aggregating `weights_delta`.
  -   `model_output`: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
  -   `client_state`: Updated local client controls.
  """
  weights_delta = attr.ib()
  control_weights_delta = attr.ib()
  client_weight = attr.ib()
  model_output = attr.ib()
  client_state = attr.ib()

class CreateClientUpdateFn():
  def __init__(self):
    self.grad_sum = None

  @tf.function
  def __call__(self, model, dataset, client_state, server_message,
               client_optimizer, use_simulation_loop):

    model_weights = _get_model_weights(model)
    initial_weights = server_message.model_weights
    mean_control_weights = server_message.mean_control_weights
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                          initial_weights)

    control_weights = tf.nest.map_structure(tf.convert_to_tensor,
                                            client_state.control_weights)
    num_batches = 0.0
    if self.grad_sum is None:
      self.grad_sum = tf.nest.map_structure(lambda x: tf.Variable(tf.zeros_like(x)), model_weights.trainable)
    tf.nest.map_structure(lambda v, t: v.assign(t), self.grad_sum, tf.nest.map_structure(tf.zeros_like, model_weights.trainable))

    def reduce_fn(state, batch):
      """Train model on local client batch."""
      num_examples, loss_sum, num_steps = state
      with tf.GradientTape() as tape:
        outputs = model.forward_pass(batch)

      grads = tape.gradient(outputs.loss, model_weights.trainable)
      corrected_grads = tf.nest.map_structure(lambda g,c,ci: g + c - ci, grads,
                                              mean_control_weights,
                                              control_weights)
      client_optimizer.apply_gradients(zip(corrected_grads, model_weights.trainable))
      if hasattr(outputs, 'num_examples'):
        batch_size = tf.cast(outputs.num_examples, dtype=tf.int32)
      else:
        batch_x, _ = _unpack_data_label(batch)
        batch_size = tf.shape(batch_x)[0]
      num_steps+=1
      num_examples += batch_size
      loss_sum += outputs.loss * tf.cast(batch_size, tf.float32)
      return num_examples, loss_sum, num_steps

    num_examples = tf.constant(0, dtype=tf.int32)
    loss_sum = tf.constant(0, dtype=tf.float32)
    num_steps = tf.constant(0, dtype=tf.float32)
    dataset_reduce_fn = _build_dataset_reduce_fn(use_simulation_loop)
    num_examples, loss_sum, num_steps = dataset_reduce_fn(
        reduce_fn, dataset, initial_state_fn=lambda: (num_examples, loss_sum, num_steps))
    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          model_weights.trainable,
                                          initial_weights.trainable)
    client_weight = tf.cast(num_examples, tf.float32)
    # Clip the norm of model delta before sending back.
    clip_norm = tf.cast(server_message.dp_clip_norm, tf.float32)
    if tf.less(tf.constant(0, tf.float32), clip_norm):
      flatten_weights_delta = tf.nest.flatten(weights_delta)
      clipped_flatten_weights_delta, _ = tf.clip_by_global_norm(
          flatten_weights_delta, clip_norm)
      weights_delta = tf.nest.pack_sequence_as(weights_delta,
                                               clipped_flatten_weights_delta)

    num_batches = tf.constant(0)
    for batch in iter(dataset):
      num_batches+=1
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      tf.nest.map_structure(lambda v, t: v.assign_add(t), self.grad_sum,
                              tape.gradient(output.loss, model_weights.trainable))

    if num_batches > 0:
      full_grad = tf.nest.map_structure(lambda a: a / tf.cast(num_batches, tf.float32),
                                        self.grad_sum)
    else:
      full_grad = tf.nest.map_structure(tf.zeros_like, model_weights.trainable)

    new_control_weights = full_grad
    control_weights_delta = tf.nest.map_structure(lambda ncw, cw: ncw - cw,
                                                  new_control_weights,
                                                  control_weights)

    return ClientOutput(
      weights_delta=weights_delta,
      control_weights_delta=control_weights_delta,
      client_weight=client_weight,
      model_output=loss_sum / client_weight,
      client_state=ClientState(
          client_id=client_state.client_id,
          control_weights=new_control_weights))

@attr.s(eq=False, frozen=True, slots=True)
class ServerState(object):
  """Structure for state on the server.

  Fields:
  -   `model`: A dictionary of model's trainable variables.
  -   `optimizer_state`: Server optimizer states.
  -   'round_num': Current round index
  -   `dp_clip_norm`: L2 norm to clip client deltas.
  -   `num_clients`: Number of public clients
  -   `mean_control_weights`: Average control weights across public clients
  """
  # Some attributes are named to be consistent with the private `ServerState` in
  # `tff.learning` to possibly use `tff.learning.build_federated_evaluation`.
  model = attr.ib()
  optimizer_state = attr.ib()
  round_num = attr.ib()
  dp_clip_norm = attr.ib()
  num_clients = attr.ib()
  mean_control_weights = attr.ib()

  @mean_control_weights.default
  def _mean_control_weights_init(self):
    return tf.nest.map_structure(tf.zeros_like, self.model.trainable)


@tf.function
def private_server_update(model, server_optimizer, server_state, weights_delta,
                  sum_control_weights_delta):
  """Updates `server_state` based on `weights_delta`.

  Args:
    model: A `KerasModelWrapper` or `tff.learning.Model`.
    server_optimizer: A `ServerOptimizerBase`.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: A nested structure of tensors holding the updates to the
      trainable variables of the model.
    sum_control_weights_delta: Sum of control variates from public clients.

  Returns:
    An updated `ServerState`.
  """
  weights_delta, has_non_finite_weight = (
      tensor_utils.zero_all_if_any_non_finite(weights_delta))
  if has_non_finite_weight > 0:
    return server_state

  # Initialize the model with the current state.
  model_weights = _get_model_weights(model)
  tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                        server_state.model)

  # Apply the update to the model, and return the updated state.
  grad = tf.nest.map_structure(lambda x: -1.0 * x, weights_delta)
  optimizer_state = server_optimizer.model_update(
      state=server_state.optimizer_state,
      weight=model_weights.trainable,
      grad=grad,
      round_idx=server_state.round_num)

  # Create a new state based on the updated model.
  return tff.structure.update_struct(
      server_state,
      model=model_weights,
      optimizer_state=optimizer_state)

@tf.function
def public_server_update(model, server_optimizer, server_state, weights_delta,
                  sum_control_weights_delta):
  """Updates `server_state` based on `weights_delta`.

  Args:
    model: A `KerasModelWrapper` or `tff.learning.Model`.
    server_optimizer: A `ServerOptimizerBase`.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: A nested structure of tensors holding the updates to the
      trainable variables of the model.
    sum_control_weights_delta: Sum of control variates from public clients.

  Returns:
    An updated `ServerState`.
  """
  weights_delta, has_non_finite_weight = (
      tensor_utils.zero_all_if_any_non_finite(weights_delta))
  if has_non_finite_weight > 0:
    return server_state

  # Initialize the model with the current state.
  model_weights = _get_model_weights(model)
  tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                        server_state.model)

  # Apply the update to the model, and return the updated state.
  grad = tf.nest.map_structure(lambda x: -1.0 * x, weights_delta)
  optimizer_state = server_optimizer.model_update(
      state=server_state.optimizer_state,
      weight=model_weights.trainable,
      grad=grad,
      round_idx=server_state.round_num)
  mean_control_weights = tf.nest.map_structure(
      lambda a,b: a + (b / tf.cast(server_state.num_clients,tf.float32)),
        server_state.mean_control_weights, sum_control_weights_delta)

  # Create a new state based on the updated model.
  return tff.structure.update_struct(
      server_state,
      model=model_weights,
      mean_control_weights=mean_control_weights)


def build_scaffold_averaging_process(
    model_fn,
    num_clients,
    dp_clip_norm=1.0,
    server_optimizer_fn=DEFAULT_SERVER_OPTIMIZER_FN,
    client_optimizer_fn=DEFAULT_CLIENT_OPTIMIZER_FN,
    use_simulation_loop=True,
    update_type='private'):
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    model_fn: A no-arg function that returns a `dp_fedavg_tf.KerasModelWrapper`.
    num_clients: Number of clients used for computing control variates.
    dp_clip_norm: if < 0, no clipping
    server_optimizer_fn: .
    client_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for client update.
    use_simulation_loop: Controls the reduce loop function for client dataset.
      Set this flag to True for performant GPU simulations.
    update_type: String for whether the process operates on private or public
      clients.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  example_model = model_fn()

  @tff.tf_computation
  def server_init_tf():
    model = model_fn()
    model_weights = _get_model_weights(model)
    optimizer = server_optimizer_fn(model_weights.trainable)
    return ServerState(
        model=model_weights,
        optimizer_state=optimizer.init_state(),
        round_num=0,
        dp_clip_norm=dp_clip_norm,
        num_clients=num_clients)

  server_state_type = server_init_tf.type_signature.result

  model_weights_type = server_state_type.model

  model_weights = _get_model_weights(example_model)
  placeholder_client_state = ClientState(
      client_id='placeholder',
      control_weights=model_weights.trainable)
  client_state_type = tff.framework.type_from_tensors(placeholder_client_state)

  @tff.tf_computation(server_state_type, model_weights_type.trainable,
                      server_state_type.mean_control_weights)
  def public_server_update_fn(server_state, model_delta, sum_control_weights_delta):
    model = model_fn()
    model_weights = _get_model_weights(model)
    optimizer = server_optimizer_fn(model_weights.trainable)
    return public_server_update(model, optimizer, server_state, model_delta,
                         sum_control_weights_delta)

  @tff.tf_computation(server_state_type, model_weights_type.trainable,
                      server_state_type.mean_control_weights)
  def private_server_update_fn(server_state, model_delta, sum_control_weights_delta):
    model = model_fn()
    model_weights = _get_model_weights(model)
    optimizer = server_optimizer_fn(model_weights.trainable)
    return private_server_update(model, optimizer, server_state, model_delta,
                         sum_control_weights_delta)

  @tff.tf_computation(server_state_type)
  def server_message_fn(server_state):
    return build_server_broadcast_message(server_state)

  server_message_type = server_message_fn.type_signature.result
  tf_dataset_type = tff.SequenceType(example_model.input_spec)

  @tff.tf_computation(tf_dataset_type, client_state_type, server_message_type)
  def client_update_fn(tf_dataset, client_state, server_message):
    model = model_fn()
    client_optimizer = client_optimizer_fn()
    client_update = CreateClientUpdateFn()
    return client_update(model, tf_dataset, client_state, server_message, client_optimizer,
                         use_simulation_loop)

  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_dataset_type = tff.type_at_clients(tf_dataset_type)

  federated_client_state_type = tff.type_at_clients(client_state_type)


  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type,
                             federated_client_state_type)
  def run_one_round_private(server_state, federated_dataset, client_states):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.data.Dataset` with placement
        `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and `tf.Tensor` of average loss.
    """
    server_message = tff.federated_map(server_message_fn, server_state)
    server_message_at_client = tff.federated_broadcast(server_message)

    client_outputs = tff.federated_map(
        client_update_fn, (federated_dataset, client_states, server_message_at_client))

    # Model deltas are equally weighted in DP.
    round_model_delta = tff.federated_mean(client_outputs.weights_delta)
    updated_client_states = client_outputs.client_state

    sum_control_weights_delta = tff.federated_sum(
        client_outputs.control_weights_delta)
    server_state = tff.federated_map(
        private_server_update_fn,
        (server_state, round_model_delta, sum_control_weights_delta))

    round_loss_metric = tff.federated_mean(client_outputs.model_output)

    return server_state, round_loss_metric, updated_client_states

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type,
                             federated_client_state_type)
  def run_one_round_public(server_state, federated_dataset, client_states):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.data.Dataset` with placement
        `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and `tf.Tensor` of average loss.
    """
    server_message = tff.federated_map(server_message_fn, server_state)
    server_message_at_client = tff.federated_broadcast(server_message)

    client_outputs = tff.federated_map(
        client_update_fn, (federated_dataset, client_states, server_message_at_client))

    # Model deltas are equally weighted in DP.
    round_model_delta = tff.federated_mean(client_outputs.weights_delta)
    updated_client_states = client_outputs.client_state

    sum_control_weights_delta = tff.federated_sum(
        client_outputs.control_weights_delta)
    server_state = tff.federated_map(
        public_server_update_fn,
        (server_state, round_model_delta, sum_control_weights_delta))

    round_loss_metric = tff.federated_mean(client_outputs.model_output)

    return server_state, round_loss_metric, updated_client_states

  @tff.federated_computation
  def server_init_tff():
    """Orchestration logic for server model initialization."""
    return tff.federated_value(server_init_tf(), tff.SERVER)

  if update_type == 'private':
    return tff.templates.IterativeProcess(
        initialize_fn=server_init_tff, next_fn=run_one_round_private)
  elif update_type == 'public':
    return tff.templates.IterativeProcess(
        initialize_fn=server_init_tff, next_fn=run_one_round_public)
