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
"""End-to-end example testing Federated Averaging."""

import collections
import functools
from absl.testing import parameterized
import numpy as np

import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy as tfp

from dp_ftrl import dp_fedavg
from dp_ftrl import optimizer_utils


def _create_test_cnn_model():
  """A simple CNN model for test."""
  data_format = 'channels_last'
  input_shape = [28, 28, 1]

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu)

  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=input_shape),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10),
      tf.keras.layers.Activation(tf.nn.softmax),
  ])

  return model


def _create_random_batch():
  return collections.OrderedDict(
      x=tf.random.uniform(tf.TensorShape([1, 28, 28, 1]), dtype=tf.float32),
      y=tf.constant(1, dtype=tf.int32, shape=[1]))


def _simple_fedavg_model_fn():
  keras_model = _create_test_cnn_model()
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
      y=tf.TensorSpec([None], tf.int32))
  return dp_fedavg.KerasModelWrapper(
      keras_model=keras_model, input_spec=input_spec, loss=loss)


def _tff_learning_model_fn():
  keras_model = _create_test_cnn_model()
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
      y=tf.TensorSpec([None], tf.int32))
  return tff.learning.from_keras_model(
      keras_model=keras_model, input_spec=input_spec, loss=loss)


MnistVariables = collections.namedtuple(
    'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')


def _create_mnist_variables():
  return MnistVariables(
      weights=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
          name='weights',
          trainable=True),
      bias=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(10)),
          name='bias',
          trainable=True),
      num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
      loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
      accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))


def _mnist_predict_on_batch(variables, batch):
  y = tf.nn.softmax(tf.matmul(batch, variables.weights) + variables.bias)
  predictions = tf.cast(tf.argmax(y, 1), tf.int32)
  return y, predictions


def _mnist_forward_pass(variables, batch):
  y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
  predictions = tf.cast(tf.argmax(y, 1), tf.int32)

  flat_labels = tf.reshape(batch['y'], [-1])
  loss = -tf.reduce_mean(
      tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(predictions, flat_labels), tf.float32))

  num_examples = tf.cast(tf.size(batch['y']), tf.float32)

  variables.num_examples.assign_add(num_examples)
  variables.loss_sum.assign_add(loss * num_examples)
  variables.accuracy_sum.assign_add(accuracy * num_examples)

  return tff.learning.BatchOutput(
      loss=loss, predictions=predictions, num_examples=num_examples)


def _get_local_mnist_metrics(variables):
  return collections.OrderedDict(
      num_examples=variables.num_examples,
      loss=variables.loss_sum / variables.num_examples,
      accuracy=variables.accuracy_sum / variables.num_examples)


@tff.federated_computation
def _aggregate_mnist_metrics_across_clients(metrics):
  return collections.OrderedDict(
      num_examples=tff.federated_sum(metrics.num_examples),
      loss=tff.federated_mean(metrics.loss, metrics.num_examples),
      accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples))


class MnistModel(tff.learning.Model):

  def __init__(self):
    self._variables = _create_mnist_variables()

  @property
  def trainable_variables(self):
    return [self._variables.weights, self._variables.bias]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def weights(self):
    return tff.learning.ModelWeights(
        trainable=self.trainable_variables,
        non_trainable=self.non_trainable_variables)

  @property
  def local_variables(self):
    return [
        self._variables.num_examples, self._variables.loss_sum,
        self._variables.accuracy_sum
    ]

  @property
  def input_spec(self):
    return collections.OrderedDict(
        x=tf.TensorSpec([None, 784], tf.float32),
        y=tf.TensorSpec([None, 1], tf.int32))

  @tf.function
  def predict_on_batch(self, batch, training=True):
    del training
    return _mnist_predict_on_batch(self._variables, batch)

  @tf.function
  def forward_pass(self, batch, training=True):
    del training
    return _mnist_forward_pass(self._variables, batch)

  @tf.function
  def report_local_outputs(self):
    return _get_local_mnist_metrics(self._variables)

  @property
  def federated_output_computation(self):
    return _aggregate_mnist_metrics_across_clients


def _create_client_data():
  emnist_batch = collections.OrderedDict(
      label=[5], pixels=np.random.rand(28, 28))

  output_types = collections.OrderedDict(label=tf.int32, pixels=tf.float32)

  output_shapes = collections.OrderedDict(
      label=tf.TensorShape([1]),
      pixels=tf.TensorShape([28, 28]),
  )

  dataset = tf.data.Dataset.from_generator(lambda: (yield emnist_batch),
                                           output_types, output_shapes)

  def client_data():
    return tff.simulation.models.mnist.keras_dataset_from_emnist(
        dataset).repeat(2).batch(2)

  return client_data


class DPFedAvgTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple_fedavg_wrapper', _simple_fedavg_model_fn),
      ('tff_learning_wrapper', _tff_learning_model_fn))
  def test_something(self, model_fn):
    it_process = dp_fedavg.build_federated_averaging_process(model_fn)
    self.assertIsInstance(it_process, tff.templates.IterativeProcess)
    federated_data_type = it_process.next.type_signature.parameter[1]
    self.assertEqual(
        str(federated_data_type),
        '{<x=float32[?,28,28,1],y=int32[?]>*}@CLIENTS')

  @parameterized.named_parameters(
      ('simple_fedavg_wrapper', _simple_fedavg_model_fn),
      ('tff_learning_wrapper', _tff_learning_model_fn))
  def test_simple_training(self, model_fn):
    it_process = dp_fedavg.build_federated_averaging_process(model_fn)
    server_state = it_process.initialize()

    def deterministic_batch():
      return collections.OrderedDict(
          x=np.ones([1, 28, 28, 1], dtype=np.float32),
          y=np.ones([1], dtype=np.int32))

    batch = tff.tf_computation(deterministic_batch)()
    federated_data = [[batch]]

    loss_list = []
    for _ in range(3):
      server_state, loss = it_process.next(server_state, federated_data)
      loss_list.append(loss)

    self.assertLess(np.mean(loss_list[1:]), loss_list[0])

  @parameterized.named_parameters(
      ('simple_fedavg_wrapper_ftrl', _simple_fedavg_model_fn,
       optimizer_utils.DPFTRLMServerOptimizer),
      ('tff_learning_wrapper_ftrl', _tff_learning_model_fn,
       optimizer_utils.DPFTRLMServerOptimizer),
      ('simple_fedavg_wrapper_sgd', _simple_fedavg_model_fn,
       optimizer_utils.DPSGDMServerOptimizer),
      ('tff_learning_wrapper_sgd', _tff_learning_model_fn,
       optimizer_utils.DPSGDMServerOptimizer),
  )
  def test_dp_momentum_training(self, model_fn, optimzer_fn, total_rounds=3):

    def server_optimzier_fn(model_weights):
      model_weight_specs = tf.nest.map_structure(
          lambda v: tf.TensorSpec(v.shape, v.dtype), model_weights)
      return optimzer_fn(
          learning_rate=1.0,
          momentum=0.9,
          noise_std=1e-5,
          model_weight_specs=model_weight_specs)

    it_process = dp_fedavg.build_federated_averaging_process(
        model_fn, server_optimizer_fn=server_optimzier_fn)
    server_state = it_process.initialize()

    def deterministic_batch():
      return collections.OrderedDict(
          x=np.ones([1, 28, 28, 1], dtype=np.float32),
          y=np.ones([1], dtype=np.int32))

    batch = tff.tf_computation(deterministic_batch)()
    federated_data = [[batch]]

    loss_list = []
    for i in range(total_rounds):
      server_state, loss = it_process.next(server_state, federated_data)
      loss_list.append(loss)
      self.assertEqual(i + 1, server_state.round_num)
      if server_state.optimizer_state is optimizer_utils.FTRLState:
        self.assertEqual(
            i + 1,
            tfp.tree_aggregation.get_step_idx(
                server_state.optimizer_state.dp_tree_state))
    self.assertLess(np.mean(loss_list[1:]), loss_list[0])

  def test_self_contained_example_custom_model(self):

    client_data = _create_client_data()
    train_data = [client_data()]

    trainer = dp_fedavg.build_federated_averaging_process(MnistModel)
    state = trainer.initialize()
    losses = []
    for _ in range(2):
      state, loss = trainer.next(state, train_data)
      losses.append(loss)
    self.assertLess(losses[1], losses[0])

  def test_keras_evaluate(self):
    keras_model = _create_test_cnn_model()
    sample_data = [
        collections.OrderedDict(
            x=np.ones([1, 28, 28, 1], dtype=np.float32),
            y=np.ones([1], dtype=np.int32))
    ]
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    metrics = dp_fedavg.keras_evaluate(keras_model, sample_data, metrics)
    accuracy = metrics[0].result()
    self.assertIsInstance(accuracy, tf.Tensor)
    self.assertBetween(accuracy, 0.0, 1.0)

  def test_tff_learning_evaluate(self):
    it_process = dp_fedavg.build_federated_averaging_process(
        _tff_learning_model_fn)
    server_state = it_process.initialize()
    sample_data = [
        collections.OrderedDict(
            x=np.ones([1, 28, 28, 1], dtype=np.float32),
            y=np.ones([1], dtype=np.int32))
    ]
    keras_model = _create_test_cnn_model()
    server_state.model.assign_weights_to(keras_model)

    sample_data = [
        collections.OrderedDict(
            x=np.ones([1, 28, 28, 1], dtype=np.float32),
            y=np.ones([1], dtype=np.int32))
    ]
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    metrics = dp_fedavg.keras_evaluate(keras_model, sample_data, metrics)
    accuracy = metrics[0].result()
    self.assertIsInstance(accuracy, tf.Tensor)
    self.assertBetween(accuracy, 0.0, 1.0)


def _server_init(model: tff.learning.Model,
                 optimizer: optimizer_utils.ServerOptimizerBase):
  """Returns initial `ServerState`."""
  return dp_fedavg.ServerState(
      model=tff.learning.ModelWeights.from_model(model),
      round_num=0,
      optimizer_state=optimizer.init_state(),
      dp_clip_norm=1000)


class ServerTest(tf.test.TestCase):

  def _assert_server_update_with_all_ones(self, model_fn):
    model = model_fn()
    state = _server_init(model, optimizer_utils.SGDServerOptimizer(0.1))
    weights_delta = tf.nest.map_structure(tf.ones_like,
                                          model.trainable_variables)

    example_optimizer = optimizer_utils.SGDServerOptimizer(0.10)
    for _ in range(2):
      state = dp_fedavg.server_update(model, example_optimizer, state,
                                      weights_delta)

    model_vars = self.evaluate(state.model)
    train_vars = model_vars.trainable
    self.assertLen(train_vars, 2)
    self.assertEqual(state.round_num, 2)
    # weights are initialized with all-zeros, weights_delta is all ones,
    # SGD learning rate is 0.1. Updating server for 2 steps.
    self.assertAllClose(train_vars, [np.ones_like(v) * 0.2 for v in train_vars])

  def test_self_contained_example_custom_model(self):
    self._assert_server_update_with_all_ones(MnistModel)


def _initialize_optimizer_vars(model, optimizer):
  """Creates optimizer variables to assign the optimizer's state."""
  model_weights = tff.learning.ModelWeights.from_model(model)
  model_delta = tf.nest.map_structure(tf.zeros_like, model_weights.trainable)
  # Create zero gradients to force an update that doesn't modify.
  # Force eagerly constructing the optimizer variables. Normally Keras lazily
  # creates the variables on first usage of the optimizer. Optimizers such as
  # Adam, Adagrad, or using momentum need to create a new set of variables shape
  # like the model weights.
  grads_and_vars = tf.nest.map_structure(
      lambda x, v: (tf.zeros_like(x), v), tf.nest.flatten(model_delta),
      tf.nest.flatten(model_weights.trainable))
  optimizer.apply_gradients(grads_and_vars)
  assert optimizer.variables()


class ClientTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('clip1000_sim', 1000, True),
      ('clip1_sim', 1, True),
      ('clip0d1_sim', 0.1, True),
      ('clip0d001_sim', 0.001, True),
      ('clip1000', 1000, False),
      ('clip1', 1, False),
      ('clip0d1', 0.1, False),
      ('clip0d001', 0.001, False),
  )
  def test_self_contained_example(self, clip_norm, simulation_flag):

    client_data = _create_client_data()

    model = MnistModel()
    optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.1)
    losses = []
    for _ in range(2):
      optimizer = optimizer_fn()
      _initialize_optimizer_vars(model, optimizer)
      server_message = dp_fedavg.BroadcastMessage(
          model_weights=tff.learning.ModelWeights.from_model(model),
          dp_clip_norm=clip_norm)
      outputs = dp_fedavg.client_update(model, client_data(), server_message,
                                        optimizer, simulation_flag)
      losses.append(outputs.model_output.numpy())
      weights_delta_norm = tf.linalg.global_norm(
          tf.nest.flatten(outputs.weights_delta))
      self.assertLessEqual(weights_delta_norm, clip_norm)

    self.assertAllEqual(int(outputs.client_weight.numpy()), 2)
    self.assertLess(losses[1], losses[0])

  def test_self_contained_example_noclip(self, clip_norm=-1):

    client_data = _create_client_data()

    model = MnistModel()
    optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.1)
    losses = []
    for _ in range(2):
      optimizer = optimizer_fn()
      _initialize_optimizer_vars(model, optimizer)
      server_message = dp_fedavg.BroadcastMessage(
          model_weights=tff.learning.ModelWeights.from_model(model),
          dp_clip_norm=clip_norm)
      outputs = dp_fedavg.client_update(
          model,
          client_data(),
          server_message,
          optimizer,
          use_simulation_loop=True)
      losses.append(outputs.model_output.numpy())

    self.assertAllEqual(int(outputs.client_weight.numpy()), 2)
    self.assertLess(losses[1], losses[0])


def _create_test_rnn_model(vocab_size: int = 6,
                           sequence_length: int = 5,
                           mask_zero: bool = True,
                           seed: int = 1) -> tf.keras.Model:
  """A simple RNN model for test."""
  initializer = tf.keras.initializers.GlorotUniform(seed=seed)
  model = tf.keras.Sequential()
  model.add(
      tf.keras.layers.Embedding(
          input_dim=vocab_size,
          input_length=sequence_length,
          output_dim=8,
          mask_zero=mask_zero,
          embeddings_initializer=initializer))
  model.add(
      tf.keras.layers.LSTM(
          units=16,
          kernel_initializer=initializer,
          recurrent_initializer='zeros',
          return_sequences=True,
          stateful=False))
  model.add(tf.keras.layers.Dense(vocab_size, kernel_initializer=initializer))
  return model


def _create_rnn_model_fn(use_tff_learning=True):

  def _rnn_model_fn():
    keras_model = _create_test_rnn_model()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    input_spec = collections.OrderedDict(
        x=tf.TensorSpec([None, 5], tf.int32),
        y=tf.TensorSpec([None, 5], tf.int32))
    if use_tff_learning:
      return tff.learning.from_keras_model(
          keras_model=keras_model, input_spec=input_spec, loss=loss)
    else:
      return dp_fedavg.KerasModelWrapper(
          keras_model=keras_model, input_spec=input_spec, loss=loss)

  return _rnn_model_fn


class RNNTest(tf.test.TestCase, parameterized.TestCase):

  def test_build_fedavg_process(self):
    it_process = dp_fedavg.build_federated_averaging_process(
        _create_rnn_model_fn())
    self.assertIsInstance(it_process, tff.templates.IterativeProcess)
    federated_type = it_process.next.type_signature.parameter
    self.assertEqual(
        str(federated_type[1]), '{<x=int32[?,5],y=int32[?,5]>*}@CLIENTS')

  def test_client_adagrad_train(self):
    it_process = dp_fedavg.build_federated_averaging_process(
        _create_rnn_model_fn(),
        client_optimizer_fn=functools.partial(
            tf.keras.optimizers.Adagrad, learning_rate=0.01))
    server_state = it_process.initialize()

    def deterministic_batch():
      return collections.OrderedDict(
          x=np.array([[0, 1, 2, 3, 4]], dtype=np.int32),
          y=np.array([[1, 2, 3, 4, 0]], dtype=np.int32))

    batch = tff.tf_computation(deterministic_batch)()
    federated_data = [[batch]]

    loss_list = []
    for _ in range(3):
      server_state, loss = it_process.next(server_state, federated_data)
      loss_list.append(loss)

    self.assertLess(np.mean(loss_list[1:]), loss_list[0])

  @parameterized.named_parameters(
      ('r5eff', 5, True, True, False),
      (
          'r5eff_reduce',
          5,
          True,
          False,
          False,
      ),
      ('r5', 5, False, True, False),
      ('r5eff_nesterov', 5, True, True, True),
  )
  def test_dpftal_training(self, total_rounds, efficient_tree, simulation_flag,
                           nesterov):

    def server_optimzier_fn(model_weights):
      model_weight_specs = tf.nest.map_structure(
          lambda v: tf.TensorSpec(v.shape, v.dtype), model_weights)
      return optimizer_utils.DPFTRLMServerOptimizer(
          learning_rate=0.1,
          momentum=0.9,
          noise_std=1e-5,
          model_weight_specs=model_weight_specs,
          efficient_tree=efficient_tree,
          use_nesterov=nesterov)

    it_process = dp_fedavg.build_federated_averaging_process(
        _create_rnn_model_fn(),
        server_optimizer_fn=server_optimzier_fn,
        use_simulation_loop=simulation_flag)
    server_state = it_process.initialize()

    def deterministic_batch():
      return collections.OrderedDict(
          x=np.array([[0, 1, 2, 3, 4]], dtype=np.int32),
          y=np.array([[1, 2, 3, 4, 0]], dtype=np.int32))

    batch = tff.tf_computation(deterministic_batch)()
    federated_data = [[batch]]

    loss_list = []
    for i in range(total_rounds):
      server_state, loss = it_process.next(server_state, federated_data)
      loss_list.append(loss)
      self.assertEqual(i + 1, server_state.round_num)
      self.assertEqual(
          i + 1,
          tfp.tree_aggregation.get_step_idx(
              server_state.optimizer_state.dp_tree_state))
    self.assertLess(np.mean(loss_list[1:]), loss_list[0])

  def test_dpftal_restart(self, total_rounds=3):

    def server_optimizer_fn(model_weights):
      model_weight_specs = tf.nest.map_structure(
          lambda v: tf.TensorSpec(v.shape, v.dtype), model_weights)
      return optimizer_utils.DPFTRLMServerOptimizer(
          learning_rate=0.1,
          momentum=0.9,
          noise_std=1e-5,
          model_weight_specs=model_weight_specs,
          efficient_tree=True,
          use_nesterov=True)

    it_process = dp_fedavg.build_federated_averaging_process(
        _create_rnn_model_fn(),
        server_optimizer_fn=server_optimizer_fn,
        use_simulation_loop=True)
    server_state = it_process.initialize()

    model = _create_rnn_model_fn()()
    model_weights = tff.learning.ModelWeights.from_model(model)
    optimizer = server_optimizer_fn(model_weights.trainable)

    def server_state_update(state):
      return tff.structure.update_struct(
          state,
          model=state.model,
          optimizer_state=optimizer.restart_dp_tree(state.model.trainable),
          round_num=state.round_num)

    def deterministic_batch():
      return collections.OrderedDict(
          x=np.array([[0, 1, 2, 3, 4]], dtype=np.int32),
          y=np.array([[1, 2, 3, 4, 0]], dtype=np.int32))

    batch = tff.tf_computation(deterministic_batch)()
    federated_data = [[batch]]

    loss_list = []
    for i in range(total_rounds):
      server_state, loss = it_process.next(server_state, federated_data)
      server_state = server_state_update(server_state)
      loss_list.append(loss)
      self.assertEqual(i + 1, server_state.round_num)
      self.assertEqual(
          0,
          tfp.tree_aggregation.get_step_idx(
              server_state.optimizer_state.dp_tree_state))
    self.assertLess(np.mean(loss_list[1:]), loss_list[0])


class DatasetReduceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('non-simulation', False, dp_fedavg._dataset_reduce_fn),
      ('simulation', True, dp_fedavg._for_iter_dataset_fn))
  def test_build_dataset_reduce_fn(self, simulation, reduce_fn):
    dataset_reduce_fn = dp_fedavg._build_dataset_reduce_fn(simulation)
    self.assertIs(dataset_reduce_fn, reduce_fn)
    ds = tf.data.Dataset.range(10, output_type=tf.int32)
    total_sum = dataset_reduce_fn(
        reduce_fn=lambda x, y: x + y, dataset=ds, initial_state_fn=lambda: 0)
    self.assertEqual(total_sum, np.int32(45))

  @parameterized.named_parameters(
      ('non-simulation', False, dp_fedavg._dataset_reduce_fn),
      ('simulation', True, dp_fedavg._for_iter_dataset_fn))
  def test_build_dataset_reduce_fn_float(self, simulation, reduce_fn):
    dataset_reduce_fn = dp_fedavg._build_dataset_reduce_fn(simulation)
    self.assertIs(dataset_reduce_fn, reduce_fn)
    ds = tf.data.Dataset.range(
        10, output_type=tf.float32).map(lambda x: 0.1 * x)
    total_sum = dataset_reduce_fn(
        reduce_fn=lambda x, y: x + y, dataset=ds, initial_state_fn=lambda: 0.)
    self.assertEqual(total_sum, np.float32(4.5))

  @parameterized.named_parameters(
      ('non-simulation', False, dp_fedavg._dataset_reduce_fn),
      ('simulation', True, dp_fedavg._for_iter_dataset_fn))
  def test_build_dataset_reduce_fn_tuple(self, simulation, reduce_fn):
    dataset_reduce_fn = dp_fedavg._build_dataset_reduce_fn(simulation)
    self.assertIs(dataset_reduce_fn, reduce_fn)
    ds = tf.data.Dataset.range(
        10, output_type=tf.float32).map(lambda x: 0.1 * x)
    total_cnt, total_sum = dataset_reduce_fn(
        reduce_fn=lambda x, y: (x[0] + 1, x[1] + y),
        dataset=ds,
        initial_state_fn=lambda: (tf.constant(0), tf.constant(0.1)))
    self.assertEqual(total_cnt, np.float32(10))
    self.assertEqual(total_sum, np.float32(4.6))


class TFFLearningDPFTRLTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('default', True, False, 0.1, 0.9),
      ('nomomentum', True, False, 0.1, 0.),
      ('nomomentum_nonoise', True, False, 0., 0.),
      ('reduce_nonoise', False, False, 0., 0.9),
      ('nesterov_nonoise', True, True, 0., 0.9),
      ('nesterov', True, True, 0.1, 0.9),
  )
  def test_dpftrl_training(self, simulation_flag, use_nesterov,
                           noise_multiplier, momentum):

    total_rounds, learning_rate, clip_norm, seed = 5, 0.1, 1., 1

    def server_optimzier_fn(model_weights):
      model_weight_specs = tf.nest.map_structure(
          lambda v: tf.TensorSpec(v.shape, v.dtype), model_weights)
      return optimizer_utils.DPFTRLMServerOptimizer(
          learning_rate=learning_rate,
          momentum=momentum,
          noise_std=clip_norm * noise_multiplier,
          model_weight_specs=model_weight_specs,
          efficient_tree=True,
          use_nesterov=use_nesterov,
          noise_seed=seed)

    it_process1 = dp_fedavg.build_federated_averaging_process(
        _create_rnn_model_fn(),
        server_optimizer_fn=server_optimzier_fn,
        dp_clip_norm=clip_norm,
        use_simulation_loop=simulation_flag)

    it_process2 = dp_fedavg.build_dpftrl_fedavg_process(
        _create_rnn_model_fn(),
        server_learning_rate=learning_rate,
        server_momentum=momentum,
        server_nesterov=use_nesterov,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        report_goal=1,
        noise_seed=seed,
        use_experimental_simulation_loop=simulation_flag)

    def deterministic_batch():
      return collections.OrderedDict(
          x=np.array([[0, 1, 2, 3, 4]], dtype=np.int32),
          y=np.array([[1, 2, 3, 4, 0]], dtype=np.int32))

    batch = tff.tf_computation(deterministic_batch)()
    federated_data = [[batch]]

    server_state1 = it_process1.initialize()
    server_state2 = it_process2.initialize()
    for _ in range(total_rounds):
      server_state1, _ = it_process1.next(server_state1, federated_data)
      server_state2, _ = it_process2.next(server_state2, federated_data)
      self.assertAllClose(
          server_state1.model.trainable,
          server_state2.model.trainable,
          rtol=1e-6)


if __name__ == '__main__':
  tf.test.main()
