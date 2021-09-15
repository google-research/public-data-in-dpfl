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
"""Server Optimizers for Federated Learning."""

import abc
import collections
from typing import Any, Collection, Dict, Optional

import attr
import tensorflow as tf
import tensorflow_privacy as tfp


def _check_momentum(m: float):
  if m < 0 or m >= 1:
    raise ValueError('Momenum should be in [0, 1), but got {}'.format(m))


class ServerOptimizerBase(metaclass=abc.ABCMeta):
  """Base class establishing interface for server optimizer."""

  @abc.abstractmethod
  def model_update(self, state: Dict[str, Any], weight: Collection[tf.Variable],
                   grad: Collection[tf.Tensor],
                   round_idx: int) -> Dict[str, Any]:
    """Returns optimizer states after modifying in-place the provided `weight`.

    Args:
      state: optimizer state, usually defined/initialized in `init_state`.
      weight: model weights to be updated in this function.
      grad: gradients to update the model weights and optimizer states.
      round_idx: round/iteration index.

    Returns:
      Updated optimizer state.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def init_state(self) -> Dict[str, Any]:
    """Returns initialized optimizer state."""
    raise NotImplementedError


class SGDServerOptimizer(ServerOptimizerBase):
  """Simple SGD Optimizer."""

  def __init__(self, learning_rate: float):
    self.lr = learning_rate

  @tf.function
  def model_update(self, state: Dict[str, Any], weight: Collection[tf.Variable],
                   grad: Collection[tf.Tensor],
                   round_idx: int) -> Dict[str, Any]:
    del round_idx, state
    tf.nest.map_structure(lambda w, g: w.assign_sub(self.lr * g), weight, grad)
    return collections.OrderedDict()

  def init_state(self):
    return collections.OrderedDict()


@attr.s(eq=False, frozen=True, slots=True)
class FTRLState(object):
  """Class defining state of the DP-FTRL optimizer.

  Attributes:
    init_weight: A Collection[tf.Tensor] defining the initial weight.
    sum_grad: A Collection[tf.Tensor] tracing the summation of gradient.
    dp_tree_state: A `tfp.tree_aggregation.TreeState` tracking the state of the
      tree aggregatin noise for the additive in DP-FTRL algorithm.
    momentum_buffer:  A Collection[tf.Tensor] tracing the velocity in the
      momentum variant. Momentum is applied to the (noised) summation of
      gradients.
  """
  init_weight = attr.ib()
  sum_grad = attr.ib()
  dp_tree_state = attr.ib()
  momentum_buffer = attr.ib()


class DPFTRLMServerOptimizer(ServerOptimizerBase):
  """Momentum FTRL Optimizer with Tree aggregation for DP noise.

  There are two options of the tree aggregation algorithm:
  the baseline method `tfp.tree_aggregation.TreeAggregator`, and the efficient
  method `tfp.tree_aggregation.EfficientTreeAggregator` , which is controlled by
  flag `efficient_tree` in the constructor.
  """

  def __init__(self,
               learning_rate: float,
               momentum: float,
               noise_std: float,
               model_weight_specs: Collection[tf.TensorSpec],
               efficient_tree: bool = True,
               use_nesterov: bool = False,
               noise_seed: Optional[int] = None):
    """Initialize the momemtum DPFTRL Optimizer."""

    _check_momentum(momentum)
    if use_nesterov and momentum == 0:
      raise ValueError('Use a positive momentum for Nesterov')

    self.lr = learning_rate
    self.momentum = momentum
    self.model_weight_specs = model_weight_specs
    self.use_nesterov = use_nesterov

    random_generator = tfp.tree_aggregation.GaussianNoiseGenerator(
        noise_std, model_weight_specs, noise_seed)

    if efficient_tree:
      self.noise_generator = tfp.tree_aggregation.EfficientTreeAggregator(
          value_generator=random_generator)
    else:
      self.noise_generator = tfp.tree_aggregation.TreeAggregator(
          value_generator=random_generator)

  @tf.function
  def model_update(self, state: FTRLState, weight: Collection[tf.Variable],
                   grad: Collection[tf.Tensor], round_idx: int) -> FTRLState:
    """Returns optimizer state after one step update."""
    init_weight, sum_grad, dp_tree_state, momentum_buffer = (
        state.init_weight, state.sum_grad, state.dp_tree_state,
        state.momentum_buffer)
    round_idx = tf.cast(round_idx, tf.int32)
    if tf.equal(round_idx, tf.constant(0, dtype=tf.int32)):
      init_weight = weight

    sum_grad = tf.nest.map_structure(tf.add, sum_grad, grad)
    cumsum_noise, dp_tree_state = self.noise_generator.get_cumsum_and_update(
        dp_tree_state)

    noised_sum_grad = tf.nest.map_structure(tf.subtract, sum_grad, cumsum_noise)
    momentum_buffer = tf.nest.map_structure(lambda v, g: self.momentum * v + g,
                                            momentum_buffer, noised_sum_grad)
    if self.use_nesterov:
      # The Nesterov implementation mimics the implementation of
      # `tf.keras.optimizers.SGD`. The forecasted weight is used to generate
      # gradient in momentum buffer (velocity).
      delta_w = tf.nest.map_structure(lambda v, g: self.momentum * v + g,
                                      momentum_buffer, noised_sum_grad)
    else:
      delta_w = momentum_buffer
    # Different from a conventional SGD step, FTRL use the initial weight w0
    # and (momementum version of) the gradient sum to update the model weight.
    tf.nest.map_structure(lambda w, w0, g: w.assign(w0 - self.lr * g), weight,
                          init_weight, delta_w)

    state = FTRLState(
        init_weight=init_weight,
        sum_grad=sum_grad,
        dp_tree_state=dp_tree_state,
        momentum_buffer=momentum_buffer)
    return state

  def _zero_state(self):
    return tf.nest.map_structure(lambda v: tf.zeros(v.shape, v.dtype),
                                 self.model_weight_specs)

  def init_state(self) -> FTRLState:
    """Returns initialized optimizer and tree aggregation states."""
    return FTRLState(
        init_weight=self._zero_state(),
        sum_grad=self._zero_state(),
        dp_tree_state=self.noise_generator.init_state(),
        momentum_buffer=self._zero_state())

  def restart_dp_tree(self, weight) -> FTRLState:
    """Returns a reinitialized state based on the current model weights."""
    return FTRLState(
        init_weight=weight,
        sum_grad=self._zero_state(),
        dp_tree_state=self.noise_generator.init_state(),
        momentum_buffer=self._zero_state())


class DPSGDMServerOptimizer(ServerOptimizerBase):
  """Momentum DPSGD Optimizer."""

  def __init__(self, learning_rate: float, momentum: float, noise_std: float,
               model_weight_specs: Collection[tf.TensorSpec]):
    """Initialize the momemtum DPSGD Optimizer."""
    self.lr = learning_rate
    self.momentum = momentum
    self.model_weight_specs = model_weight_specs

    self.noise_std = noise_std
    # TODO(b/177243233): possibly add the state of noise generator to the state
    # of optimzier. TFF now rely on the non-deterministic initialization of the
    # random noise generator and the states/variables have to be created inside
    # the TFF computation.
    self.random_generator = tf.random.Generator.from_non_deterministic_state()

  def _noise_fn(self):
    """Returns random noise to be added for differential privacy."""

    def noise_tensor(spec):
      noise = self.random_generator.normal(spec.shape, stddev=self.noise_std)
      # TODO(b/177259859): reshape because the shape of the noise could have
      # None/? that fails TFF type check.
      noise = tf.reshape(noise, spec.shape)
      return noise

    return tf.nest.map_structure(noise_tensor, self.model_weight_specs)

  @tf.function
  def model_update(self, state: Dict[str, Any], weight: Collection[tf.Variable],
                   grad: Collection[tf.Tensor],
                   round_idx: int) -> Dict[str, Any]:
    """Returns optimizer state after one step update."""
    del round_idx
    momentum_buffer = state['momentum_buffer']

    momentum_buffer = tf.nest.map_structure(
        lambda m, g, n: self.momentum * m + g + n, momentum_buffer, grad,
        self._noise_fn())
    tf.nest.map_structure(lambda w, v: w.assign_sub(self.lr * v), weight,
                          momentum_buffer)
    state = collections.OrderedDict(momentum_buffer=momentum_buffer)
    return state

  def init_state(self) -> Dict[str, Any]:
    """Returns initialized momentum buffer."""

    def _zero_state():
      return tf.nest.map_structure(lambda v: tf.zeros(v.shape, v.dtype),
                                   self.model_weight_specs)

    return collections.OrderedDict(momentum_buffer=_zero_state())
