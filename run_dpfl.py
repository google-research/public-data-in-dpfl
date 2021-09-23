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
"""
Trains and evaluates NWP model on a variety of private/public data
experiments.
"""

import collections
import functools
import random
from typing import List, Tuple, Optional

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff
import sys

import language_models as models
import metrics
import data_word_prediction as nwp_dataset
import scaffold_v2
import mime
import mimelite
import alternating_loop
import warmstart_loop
import scaffold_loop
import mime_loop
import mimelite_loop
import mirror_descent
import mirror_descent_convex
import mirror_descent_loop

from dp_ftrl import dp_fedavg
from dp_ftrl import optimizer_utils
from dp_ftrl import training_loop
from utils import keras_metrics
from utils.datasets import stackoverflow_word_prediction as stackoverflow_dataset



IRRELEVANT_FLAGS = frozenset(iter(flags.FLAGS))

flags.DEFINE_string(
    'experiment_name', 'stackoverflow', 'The name of this experiment. Will be'
    'append to  --root_output_dir to separate experiment results.')
flags.DEFINE_enum('experiment_type', 'private', [
    'private', 'public', 'public_SO', 'warmstart', 'warmstart_SO',
    'alternating', 'alternating_SO', 'scaffold', 'scaffold_SO',
    'alternating_warmstart', 'alternating_warmstart_SO', 'mime', 'mime_SO',
    'mimelite', 'mimelite_SO', 'mimelite_warmstart','mimelite_warmstart_SO',
    'mime_warmstart','mime_warmstart_SO', 'stackoverflow_SGD',
    'mirror_descent','mirror_descent_warmstart','mirror_descent_SO','mirror_descent_warmstart_SO',
    'mirror_descent_convex','mirror_descent_convex_warmstart','mirror_descent_convex_SO','mirror_descent_convex_warmstart_SO'
], 'Experiment type that we wish to run.')
flags.DEFINE_string('dataset', 'stackoverflow', 'Name of dataset for training,')
flags.DEFINE_string('root_output_dir', '/tmp/dpftrl/public_dpfl',
                    'Root directory for writing experiment output.')
flags.DEFINE_integer('rounds_per_checkpoint', 20,
                     'How often to checkpoint the global model.')
flags.DEFINE_integer(
    'rounds_per_eval', 20,
    'How often to evaluate the global model on the validation dataset.')
flags.DEFINE_integer('clients_per_thread', 1, 'TFF executor configuration.')

# Training
flags.DEFINE_integer('clients_per_round', 100,
                     'How many clients to sample per round.')
flags.DEFINE_integer('client_epochs_per_round', 1,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('client_batch_size', 16, 'Batch size used on the client.')
flags.DEFINE_integer('total_rounds', 10, 'Number of total training rounds.')
flags.DEFINE_integer(
    'total_epochs', None,
    'If not None, use shuffling of clients instead of random sampling.')

# Optimizer
flags.DEFINE_enum('client_optimizer', 'sgd', ['sgd', 'sgdm'], 'Client optimzier')
flags.DEFINE_enum(
    'server_optimizer', 'sgd',
    ['sgd', 'sgdm', 'adam', 'dpftrl', 'dpsgd', 'dpsgdm', 'dpftrlm'],
    'Server optimizer in federated optimizaiotn.')
flags.DEFINE_float('client_lr', 0.02, 'Client learning rate.')
flags.DEFINE_float('server_lr', 1.0, 'Server learning rate.')
flags.DEFINE_float('server_momentum', 0.9, 'Server momentum for SGDM.')
flags.DEFINE_boolean(
    'use_tff_learning', False,
    'Boolean indicating whether to use `tff.learning` to build iterative'
    'process for training. If True, server optimizer has to be `dpftrlm`')

# Differential privacy
flags.DEFINE_float('clip_norm', 1.0, 'Clip L2 norm.')
flags.DEFINE_float('noise_multiplier', 0.01,
                   'Noise multiplier for DP algorithm.')

# Data
flags.DEFINE_integer('sequence_length', 20, 'Max sequence length to use.')
flags.DEFINE_integer('max_elements_per_user', 256, 'Max number of training '
                     'sentences to use per user.')
flags.DEFINE_integer(
    'num_validation_examples', 10000, 'Number of examples '
    'to use from test set for per-round validation.')

# Model
flags.DEFINE_integer('vocab_size', 10000, 'Size of vocab to use.')
flags.DEFINE_integer('num_oov_buckets', 1,
                     'Number of out of vocabulary buckets.')
flags.DEFINE_integer('embedding_size', 96,
                     'Dimension of word embedding to use.')
flags.DEFINE_integer('latent_size', 670,
                     'Dimension of latent size to use in recurrent cell')
flags.DEFINE_integer('num_layers', 1,
                     'Number of stacked recurrent layers to use.')
flags.DEFINE_enum('lstm_cell', 'LSTM', ['LSTM', 'CIFG'],
                  'Cell type for recurrent LM.')
flags.DEFINE_boolean(
    'shared_embedding', False,
    'Boolean indicating whether to tie input and output embeddings.')
flags.DEFINE_integer(
    'consecutive_private_rounds', 10,
    'Number of consecutive private updates in alternating.')
flags.DEFINE_integer(
    'consecutive_public_rounds', 1,
    'Number of consecutive public updates in alternating.')
flags.DEFINE_integer('private_round_size', 100,
                     'Number of clients in a private round')
flags.DEFINE_integer('public_round_size', 20,
                     'Number of clients in a public round')
flags.DEFINE_boolean('warmstart', False,
                     'Boolean indicating whether to warm start the model')
flags.DEFINE_string('private_server_optimizer','dpsgdm',
                    'Optimizer for private updates.')
flags.DEFINE_string('public_server_optimizer', 'sgdm',
                    'Optimizer for public updates.')
flags.DEFINE_integer('private_client_batch_size', 16, 'Batch size used on the private clients.')
flags.DEFINE_integer('public_client_batch_size', 16, 'Batch size used on the public clients.')
flags.DEFINE_integer('private_clients_per_round', 100,
                     'How many private clients to sample per round.')
flags.DEFINE_integer('public_clients_per_round', 100,
                     'How many private clients to sample per round.')
flags.DEFINE_string('warmstart_file', None,
                    'File path for checkpoint to start finetuning')
flags.DEFINE_string('private_dataset','stackoverflow',
                    'Optimizer for private updates.')
flags.DEFINE_string('public_dataset', 'stackoverflow',
                    'Optimizer for public updates.')
flags.DEFINE_integer('samples_in_public_gradient',10000,
                     'Number of samples to use for full public gradient in SVRG')
flags.DEFINE_integer('update_public_gradient_frequency',100,
                     'Epoch length for SVRG')
flags.DEFINE_enum('public_clipping_strategy', 'individual', ['individual','average'],
                  'Clipping strategy for the public updates.')
flags.DEFINE_boolean('restart_optimizer', False,
                     'Whether to restart the optimizer during alternating.')
flags.DEFINE_float('public_data_percentage', 1.00, 'Percentage of original clients to use in warmstarting')
flags.DEFINE_integer('update_private_gradient_frequency',10,
                     'Epoch length for MD')
flags.DEFINE_float('private_lr', 1.00, 'Learning rate for private update in MD update.')

HPARAM_FLAGS = [f for f in flags.FLAGS if f not in IRRELEVANT_FLAGS]
FLAGS = flags.FLAGS


def _get_metrics(vocab_size, num_oov_buckets):
  """Metrics for stackoverflow dataset."""
  special_tokens = nwp_dataset.get_special_tokens(
      vocab_size, num_oov_buckets)
  pad_token = special_tokens.pad
  oov_tokens = special_tokens.oov
  eos_token = special_tokens.eos
  return [
      keras_metrics.MaskedCategoricalAccuracy(
          name='accuracy_with_oov', masked_tokens=[pad_token]),
      keras_metrics.MaskedCategoricalAccuracy(
          name='accuracy_no_oov', masked_tokens=[pad_token] + oov_tokens),
      keras_metrics.MaskedCategoricalAccuracy(
          name='accuracy_no_oov_or_eos',
          masked_tokens=[pad_token, eos_token] + oov_tokens),
      metrics.Perplexity(name='perplexity', padding_variable=pad_token),
      tf.keras.metrics.SparseCategoricalCrossentropy(
          name='loss', from_logits=True)
  ]


def _preprocess_data(data_name, vocab_size, num_oov_buckets, sequence_length,
                     num_validation_examples, client_batch_size,
                     client_epochs_per_round, max_elements_per_user):
  if data_name == 'stackoverflow':
    train_clientdata, _, test_clientdata = (
        tff.simulation.datasets.stackoverflow.load_data(cache_dir='/scratch/hdd001/home/vinithms/data_dpfl/'))
    dataset_vocab = stackoverflow_dataset.create_vocab(vocab_size)

    base_test_dataset = test_clientdata.create_tf_dataset_from_all_clients()
    preprocess_val_and_test = stackoverflow_dataset.create_preprocess_fn(
        vocab=dataset_vocab,
        num_oov_buckets=num_oov_buckets,
        client_batch_size=128,
        client_epochs_per_round=client_epochs_per_round,
        max_sequence_length=sequence_length,
        max_elements_per_client=-1,
        max_shuffle_buffer_size=1)
    test_set = preprocess_val_and_test(
        base_test_dataset.skip(num_validation_examples))
    validation_set = preprocess_val_and_test(
        base_test_dataset.take(num_validation_examples))

    train_dataset_preprocess_comp =stackoverflow_dataset.create_preprocess_fn(
        vocab=dataset_vocab,
        num_oov_buckets=num_oov_buckets,
        client_batch_size=client_batch_size,
        client_epochs_per_round=client_epochs_per_round,
        max_sequence_length=sequence_length,
        max_elements_per_client=max_elements_per_user,
        max_shuffle_buffer_size=max_elements_per_user)
        # dataset='stackoverflow')
  elif data_name == 'stackoverflow_private':
    train_clientdata, _, test_clientdata = (
        tff.simulation.datasets.stackoverflow.load_data(cache_dir='/scratch/gobi2/vinithms/public-data-in-dpfl/data/'))
    dataset_vocab = nwp_dataset.create_vocab(vocab_size)

    base_test_dataset = test_clientdata.create_tf_dataset_from_all_clients()
    preprocess_val_and_test = nwp_dataset.create_preprocess_fn(
        vocab=dataset_vocab,
        num_oov_buckets=num_oov_buckets,
        client_batch_size=128,
        client_epochs_per_round=client_epochs_per_round,
        max_sequence_length=sequence_length,
        max_elements_per_client=-1,
        max_shuffle_buffer_size=1,
        dataset='stackoverflow')
    test_set = preprocess_val_and_test(
        base_test_dataset.skip(num_validation_examples))
    validation_set = preprocess_val_and_test(
        base_test_dataset.take(num_validation_examples))

    train_dataset_preprocess_comp = nwp_dataset.create_preprocess_fn(
        vocab=dataset_vocab,
        num_oov_buckets=num_oov_buckets,
        client_batch_size=client_batch_size,
        client_epochs_per_round=client_epochs_per_round,
        max_sequence_length=sequence_length,
        max_elements_per_client=max_elements_per_user,
        max_shuffle_buffer_size=max_elements_per_user,
        dataset='stackoverflow')

  elif data_name == 'stackoverflow_public':
    _, train_clientdata, test_clientdata = (
        tff.simulation.datasets.stackoverflow.load_data(cache_dir='/scratch/gobi2/vinithms/public-data-in-dpfl/data/'))
    # _, _, test_clientdata = (
    #     tff.simulation.datasets.stackoverflow.load_data())
    dataset_vocab = nwp_dataset.create_vocab(vocab_size)

    base_test_dataset = test_clientdata.create_tf_dataset_from_all_clients()
    preprocess_val_and_test = nwp_dataset.create_preprocess_fn(
        vocab=dataset_vocab,
        num_oov_buckets=num_oov_buckets,
        client_batch_size=128,
        client_epochs_per_round=client_epochs_per_round,
        max_sequence_length=sequence_length,
        max_elements_per_client=-1,
        max_shuffle_buffer_size=1,
        dataset='stackoverflow')
    test_set = preprocess_val_and_test(
        base_test_dataset.skip(num_validation_examples))
    validation_set = preprocess_val_and_test(
        base_test_dataset.take(num_validation_examples))

    train_dataset_preprocess_comp = nwp_dataset.create_preprocess_fn(
        vocab=dataset_vocab,
        num_oov_buckets=num_oov_buckets,
        client_batch_size=client_batch_size,
        client_epochs_per_round=client_epochs_per_round,
        max_sequence_length=sequence_length,
        max_elements_per_client=max_elements_per_user,
        max_shuffle_buffer_size=max_elements_per_user,
        dataset='stackoverflow')
  else:
    raise ValueError('Unknown dataset name {}'.format(data_name))

  @tff.tf_computation(tf.string)
  def train_dataset_computation(client_id):
    client_train_data = train_clientdata.dataset_computation(client_id)
    return train_dataset_preprocess_comp(client_train_data)

  return train_dataset_computation, train_clientdata, validation_set, test_set

def _preprocess_scaffold_data(vocab_size, num_oov_buckets, sequence_length,
                     num_validation_examples, client_batch_size,
                     client_epochs_per_round, max_elements_per_user):

    train_clientdata_private, _, test_clientdata = (
      tff.simulation.datasets.stackoverflow.load_data())
    dataset_vocab =stackoverflow_dataset.create_vocab(vocab_size)

    base_test_dataset = test_clientdata.create_tf_dataset_from_all_clients()
    preprocess_val_and_test = nwp_dataset.create_preprocess_fn(
        vocab=dataset_vocab,
        num_oov_buckets=num_oov_buckets,
        client_batch_size=128,
        client_epochs_per_round=client_epochs_per_round,
        max_sequence_length=sequence_length,
        max_elements_per_client=-1,
        max_shuffle_buffer_size=1,
        dataset='stackoverflow')
    test_set = preprocess_val_and_test(
        base_test_dataset.skip(num_validation_examples))
    validation_set = preprocess_val_and_test(
        base_test_dataset.take(num_validation_examples))

    train_dataset_preprocess_comp_private = nwp_dataset.create_preprocess_fn(
        vocab=dataset_vocab,
        num_oov_buckets=num_oov_buckets,
        client_batch_size=client_batch_size,
        client_epochs_per_round=client_epochs_per_round,
        max_sequence_length=sequence_length,
        max_elements_per_client=max_elements_per_user,
        max_shuffle_buffer_size=max_elements_per_user,
        dataset='stackoverflow')

    if FLAGS.experiment_type == 'scaffold_SO':
      _, train_clientdata_public, _ = (
        tff.simulation.datasets.stackoverflow.load_data())
      dataset_vocab = nwp_dataset.create_vocab(vocab_size)

      train_dataset_preprocess_comp_public = nwp_dataset.create_preprocess_fn(
          vocab=dataset_vocab,
          num_oov_buckets=num_oov_buckets,
          client_batch_size=client_batch_size,
          client_epochs_per_round=client_epochs_per_round,
          max_sequence_length=sequence_length,
          max_elements_per_client=max_elements_per_user,
          max_shuffle_buffer_size=max_elements_per_user,
          dataset='stackoverflow')

    @tff.tf_computation(tf.string)
    def train_dataset_computation_private(client_id):
      logging.info("Computing private client")
      client_train_data_private = train_clientdata_private.dataset_computation(client_id)
      return train_dataset_preprocess_comp_private(client_train_data_private)

    @tff.tf_computation(tf.string)
    def train_dataset_computation_public(client_id):
      logging.info("Computing public client")
      client_train_data_public = train_clientdata_public.dataset_computation(client_id)
      return train_dataset_preprocess_comp_public(client_train_data_public)

    logging.info("Returning from scaffold data process")
    return train_dataset_computation_private,train_dataset_computation_public, train_clientdata_private, train_clientdata_public, validation_set, test_set

def _preprocess_mime_data(vocab_size, num_oov_buckets, sequence_length,
                     num_validation_examples, client_batch_size,
                     client_epochs_per_round, max_elements_per_user):

    train_clientdata_private, _, test_clientdata = (
      tff.simulation.datasets.stackoverflow.load_data())

    dataset_vocab =stackoverflow_dataset.create_vocab(vocab_size)

    base_test_dataset = test_clientdata.create_tf_dataset_from_all_clients()
    preprocess_val_and_test = nwp_dataset.create_preprocess_fn(
        vocab=dataset_vocab,
        num_oov_buckets=num_oov_buckets,
        client_batch_size=128,
        client_epochs_per_round=client_epochs_per_round,
        max_sequence_length=sequence_length,
        max_elements_per_client=-1,
        max_shuffle_buffer_size=1,
        dataset='stackoverflow')
    test_set = preprocess_val_and_test(
        base_test_dataset.skip(num_validation_examples))
    validation_set = preprocess_val_and_test(
        base_test_dataset.take(num_validation_examples))

    train_dataset_preprocess_comp_private = nwp_dataset.create_preprocess_fn(
        vocab=dataset_vocab,
        num_oov_buckets=num_oov_buckets,
        client_batch_size=client_batch_size,
        client_epochs_per_round=client_epochs_per_round,
        max_sequence_length=sequence_length,
        max_elements_per_client=max_elements_per_user,
        max_shuffle_buffer_size=max_elements_per_user,
        dataset='stackoverflow')

    if FLAGS.experiment_type == 'mime_SO' or FLAGS.experiment_type == 'mime_warmstart_SO':
      _, train_clientdata_public, _ = (
        tff.simulation.datasets.stackoverflow.load_data())
      
      dataset_vocab = nwp_dataset.create_vocab(vocab_size)

      train_dataset_preprocess_comp_public = nwp_dataset.create_preprocess_fn(
          vocab=dataset_vocab,
          num_oov_buckets=num_oov_buckets,
          client_batch_size=client_batch_size,
          client_epochs_per_round=client_epochs_per_round,
          max_sequence_length=sequence_length,
          max_elements_per_client=max_elements_per_user,
          max_shuffle_buffer_size=max_elements_per_user,
          dataset='stackoverflow')

    @tff.tf_computation(tf.string)
    def train_dataset_computation_private(client_id):
      logging.info("Computing private client")
      client_train_data_private = train_clientdata_private.dataset_computation(client_id)
      return train_dataset_preprocess_comp_private(client_train_data_private)

    @tff.tf_computation(tf.string)
    def train_dataset_computation_public(client_id):
      logging.info("Computing public client")
      client_train_data_public = train_clientdata_public.dataset_computation(client_id)
      return train_dataset_preprocess_comp_public(client_train_data_public)

    return train_dataset_computation_private,train_dataset_computation_public, train_clientdata_private, train_clientdata_public, validation_set, test_set

def _preprocess_mimelite_data(vocab_size, num_oov_buckets, sequence_length,
                     num_validation_examples, client_batch_size,
                     client_epochs_per_round, max_elements_per_user):

    train_clientdata_private, _, test_clientdata = (
      tff.simulation.datasets.stackoverflow.load_data())
    dataset_vocab =stackoverflow_dataset.create_vocab(vocab_size)

    base_test_dataset = test_clientdata.create_tf_dataset_from_all_clients()
    preprocess_val_and_test = nwp_dataset.create_preprocess_fn(
        vocab=dataset_vocab,
        num_oov_buckets=num_oov_buckets,
        client_batch_size=128,
        client_epochs_per_round=client_epochs_per_round,
        max_sequence_length=sequence_length,
        max_elements_per_client=-1,
        max_shuffle_buffer_size=1,
        dataset='stackoverflow')
    test_set = preprocess_val_and_test(
        base_test_dataset.skip(num_validation_examples))
    validation_set = preprocess_val_and_test(
        base_test_dataset.take(num_validation_examples))

    train_dataset_preprocess_comp_private = nwp_dataset.create_preprocess_fn(
        vocab=dataset_vocab,
        num_oov_buckets=num_oov_buckets,
        client_batch_size=client_batch_size,
        client_epochs_per_round=client_epochs_per_round,
        max_sequence_length=sequence_length,
        max_elements_per_client=max_elements_per_user,
        max_shuffle_buffer_size=max_elements_per_user,
        dataset='stackoverflow')

    if FLAGS.experiment_type == 'mimelite_SO' or FLAGS.experiment_type == 'mimelite_warmstart_SO':
       _, train_clientdata_public, test_clientdata = (
         tff.simulation.datasets.stackoverflow.load_data())
       dataset_vocab = nwp_dataset.create_vocab(vocab_size)

       train_dataset_preprocess_comp_public = nwp_dataset.create_preprocess_fn(
          vocab=dataset_vocab,
          num_oov_buckets=num_oov_buckets,
          client_batch_size=client_batch_size,
          client_epochs_per_round=client_epochs_per_round,
          max_sequence_length=sequence_length,
          max_elements_per_client=max_elements_per_user,
          max_shuffle_buffer_size=max_elements_per_user,
          dataset='stackoverflow')

    @tff.tf_computation(tf.string)
    def train_dataset_computation_private(client_id):
      logging.info("Computing private client")
      client_train_data_private = train_clientdata_private.dataset_computation(client_id)
      return train_dataset_preprocess_comp_private(client_train_data_private)

    @tff.tf_computation(tf.string)
    def train_dataset_computation_public(client_id):
      logging.info("Computing public client")
      client_train_data_public = train_clientdata_public.dataset_computation(client_id)
      return train_dataset_preprocess_comp_public(client_train_data_public)

    logging.info("Returning from mimelite data process")
    return train_dataset_computation_private,train_dataset_computation_public, train_clientdata_private, train_clientdata_public, validation_set, test_set


def _server_optimizer_fn(model_weights, name, learning_rate, noise_std):
  """Returns server optimizer."""
  model_weight_specs = tf.nest.map_structure(
      lambda v: tf.TensorSpec(v.shape, v.dtype), model_weights)
  if name == 'sgd':
    return optimizer_utils.SGDServerOptimizer(learning_rate)
  elif name == 'sgdm':
    return optimizer_utils.DPSGDMServerOptimizer(
        learning_rate,
        momentum=FLAGS.server_momentum,
        noise_std=0,
        model_weight_specs=model_weight_specs)
  elif name == 'dpftrl':
    return optimizer_utils.DPFTRLMServerOptimizer(
        learning_rate,
        momentum=0,
        noise_std=noise_std,
        model_weight_specs=model_weight_specs)
  elif name == 'dpsgd':
    return optimizer_utils.DPSGDMServerOptimizer(
        learning_rate,
        momentum=0,
        noise_std=noise_std,
        model_weight_specs=model_weight_specs)
  elif name == 'dpsgdm':
    return optimizer_utils.DPSGDMServerOptimizer(
        learning_rate,
        momentum=FLAGS.server_momentum,
        noise_std=noise_std,
        model_weight_specs=model_weight_specs)
  elif name == 'dpftrlm':
    return optimizer_utils.DPFTRLMServerOptimizer(
        learning_rate,
        momentum=FLAGS.server_momentum,
        noise_std=noise_std,
        model_weight_specs=model_weight_specs)
  else:
    raise ValueError('Unknown server optimizer name {}'.format(name))


def _build_server_state_epoch_update_fn(server_optimizer_name, model_fn,
                                        server_optimizer_fn):
  """Build server update function: tree restart for FTRL."""
  if server_optimizer_name == 'dpftrl' or server_optimizer_name == 'dpftrlm':
    # A server optimzier is built to get a new state to restart the optimizer.
    # A model is built to initialize the optimizer because the optimizer state
    # depends on the shape of the model weights. The model and optimizer are
    # only constructed once.
    model = model_fn()
    optimizer = server_optimizer_fn(model.weights.trainable)

    def server_state_update(state):
      return tff.structure.update_struct(
          state,
          model=state.model,
          optimizer_state=optimizer.restart_dp_tree(state.model.trainable),
          round_num=state.round_num)

    return server_state_update
  else:
    return None


def _client_optimizer_fn(name, learning_rate, momentum=0.0):
  if name == 'sgd':
    return tf.keras.optimizers.SGD(learning_rate=learning_rate)
  elif name == 'sgdm':
    return tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum)
  else:
    raise ValueError('Unknown client optimizer name {}'.format(name))


def _sample_client_ids(
    num_clients: int,
    client_data: tff.simulation.datasets.ClientData,
    round_num: int,
    epoch: int,
) -> Tuple[List, int]:  # pylint: disable=g-bare-generic
  """Returns a random subset of client ids."""
  del round_num  # Unused.
  return random.sample(client_data.client_ids, num_clients), epoch

def _sample_public_client_ids(
    num_clients: int,
    client_data: List,
    round_num: int,
    epoch: int,
) -> Tuple[List, int]:  # pylint: disable=g-bare-generic
  """Returns a random subset of client ids."""
  del round_num  # Unused.
  return random.sample(client_data, num_clients), epoch


def _build_custom_model_and_process(input_spec, test_metrics,
                                    server_optimizer: str = 'sgd',
                                    update_type: str = 'individual',
                                    public_dataset: Optional[tf.data.Dataset] = None):
  """Build customized iterative process."""

  if FLAGS.experiment_type != 'alternating' and FLAGS.experiment_type != 'alternating_SO' and FLAGS.experiment_type != 'alternating_warmstart' and FLAGS.experiment_type != 'alternating_warmstart_SO':
    server_optimizer = FLAGS.server_optimizer
    logging.info("Following server optimizer flag.")

  def tff_model_fn():
    keras_model = models.create_recurrent_model(
        vocab_size=FLAGS.vocab_size,
        embedding_size=FLAGS.embedding_size,
        latent_size=FLAGS.latent_size,
        num_layers=FLAGS.num_layers,
        shared_embedding=FLAGS.shared_embedding,
        cell_type=FLAGS.lstm_cell)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return dp_fedavg.KerasModelWrapper(keras_model, input_spec, loss)

  noise_std = FLAGS.clip_norm * FLAGS.noise_multiplier / float(
      FLAGS.clients_per_round)
  server_optimizer_fn = functools.partial(
      _server_optimizer_fn,
      name=server_optimizer,
      learning_rate=FLAGS.server_lr,
      noise_std=noise_std)
  client_optimizer_fn = functools.partial(
      _client_optimizer_fn,
      name=FLAGS.client_optimizer,
      learning_rate=FLAGS.client_lr)
  if update_type == 'individual':
    iterative_process = dp_fedavg.build_federated_averaging_process(
        tff_model_fn,
        dp_clip_norm=FLAGS.clip_norm,
        server_optimizer_fn=server_optimizer_fn,
        client_optimizer_fn=client_optimizer_fn)
  else:
    raise ValueError("Unknown update_type %s".format(update_type))

  model = tff_model_fn()

  def evaluate_fn(model_weights, dataset):
    model.from_weights(model_weights)
    metrics = dp_fedavg.keras_evaluate(model.keras_model, dataset, test_metrics)
    return collections.OrderedDict(
        (metric.name, metric.result().numpy()) for metric in metrics)

  server_state_update_fn = _build_server_state_epoch_update_fn(
      server_optimizer, tff_model_fn, server_optimizer_fn)
  return iterative_process, evaluate_fn, server_state_update_fn

def _build_scaffold_model_and_process(input_spec, test_metrics, server_optimizer, update_type='private'):
  """Build scaffold iterative process."""

  def tff_model_fn():
    keras_model = models.create_recurrent_model(
        vocab_size=FLAGS.vocab_size,
        embedding_size=FLAGS.embedding_size,
        latent_size=FLAGS.latent_size,
        num_layers=FLAGS.num_layers,
        shared_embedding=FLAGS.shared_embedding,
        cell_type=FLAGS.lstm_cell)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return dp_fedavg.KerasModelWrapper(keras_model, input_spec, loss)

  noise_std = FLAGS.clip_norm * FLAGS.noise_multiplier / float(
      FLAGS.clients_per_round)
  server_optimizer_fn = functools.partial(
      _server_optimizer_fn,
      name=server_optimizer,
      learning_rate=FLAGS.server_lr,
      noise_std=noise_std)
  client_optimizer_fn = functools.partial(
      _client_optimizer_fn,
      name=FLAGS.client_optimizer,
      learning_rate=FLAGS.client_lr)
  iterative_process = scaffold_v2.build_scaffold_averaging_process(
      tff_model_fn,
      FLAGS.public_round_size,
      update_type=update_type,
      dp_clip_norm=FLAGS.clip_norm,
      server_optimizer_fn=server_optimizer_fn,
      client_optimizer_fn=client_optimizer_fn)

  model = tff_model_fn()

  def client_init_fn(client_id):
    control_weights = tf.nest.map_structure(tf.zeros_like,
                                            model.weights.trainable)
    return scaffold_v2.ClientState(
        client_id=client_id, control_weights=control_weights)

  def evaluate_fn(model_weights, dataset):
    model.from_weights(model_weights)
    metrics = dp_fedavg.keras_evaluate(model.keras_model, dataset, test_metrics)
    return collections.OrderedDict(
        (metric.name, metric.result().numpy()) for metric in metrics)

  server_state_update_fn = _build_server_state_epoch_update_fn(
      FLAGS.server_optimizer, tff_model_fn, server_optimizer_fn)
  return iterative_process, evaluate_fn, server_state_update_fn, client_init_fn

def _build_mime_model_and_process(input_spec, test_metrics, server_optimizer, update_type='private'):
  """Build mime iterative process."""

  def tff_model_fn():
    keras_model = models.create_recurrent_model(
        vocab_size=FLAGS.vocab_size,
        embedding_size=FLAGS.embedding_size,
        latent_size=FLAGS.latent_size,
        num_layers=FLAGS.num_layers,
        shared_embedding=FLAGS.shared_embedding,
        cell_type=FLAGS.lstm_cell)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return dp_fedavg.KerasModelWrapper(keras_model, input_spec, loss)

  noise_std = FLAGS.clip_norm * FLAGS.noise_multiplier / float(
      FLAGS.clients_per_round)
  optimizer_fn = tf.keras.optimizers.SGD
  iterative_process = mime.build_averaging_process(
      tff_model_fn,
      optimizer_fn=optimizer_fn,
      update_type=update_type,
      base_lr=FLAGS.client_lr,
      server_lr=FLAGS.server_lr,
      server_momentum=FLAGS.server_momentum,
      dp_clip_norm=FLAGS.clip_norm,
      dp_noise_std= FLAGS.clip_norm * FLAGS.noise_multiplier / float(
      FLAGS.private_round_size))
  model = tff_model_fn()

  def evaluate_fn(model_weights, dataset):
    model.from_weights(model_weights)
    metrics = dp_fedavg.keras_evaluate(model.keras_model, dataset, test_metrics)
    return collections.OrderedDict(
        (metric.name, metric.result().numpy()) for metric in metrics)

  server_state_update_fn = _build_server_state_epoch_update_fn(
      FLAGS.server_optimizer, tff_model_fn, optimizer_fn)
  return iterative_process, evaluate_fn, server_state_update_fn

def _build_mimelite_model_and_process(input_spec, test_metrics, server_optimizer, update_type='private'):
  """Build mimelite iterative process."""

  def tff_model_fn():
    keras_model = models.create_recurrent_model(
        vocab_size=FLAGS.vocab_size,
        embedding_size=FLAGS.embedding_size,
        latent_size=FLAGS.latent_size,
        num_layers=FLAGS.num_layers,
        shared_embedding=FLAGS.shared_embedding,
        cell_type=FLAGS.lstm_cell)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return dp_fedavg.KerasModelWrapper(keras_model, input_spec, loss)

  noise_std = FLAGS.clip_norm * FLAGS.noise_multiplier / float(
      FLAGS.clients_per_round)

  optimizer_fn = tf.keras.optimizers.SGD
  iterative_process = mimelite.build_averaging_process(
      tff_model_fn,
      optimizer_fn=optimizer_fn,
      update_type=update_type,
      base_lr=FLAGS.client_lr,
      server_lr=FLAGS.server_lr,
      server_momentum=FLAGS.server_momentum,
      dp_clip_norm=FLAGS.clip_norm,
      dp_noise_std= FLAGS.clip_norm * FLAGS.noise_multiplier / float(
      FLAGS.private_round_size))

  model = tff_model_fn()

  def evaluate_fn(model_weights, dataset):
    model.from_weights(model_weights)
    metrics = dp_fedavg.keras_evaluate(model.keras_model, dataset, test_metrics)
    return collections.OrderedDict(
        (metric.name, metric.result().numpy()) for metric in metrics)

  server_state_update_fn = _build_server_state_epoch_update_fn(
      FLAGS.server_optimizer, tff_model_fn, optimizer_fn)
  return iterative_process, evaluate_fn, server_state_update_fn

def _build_mirror_descent_model_and_process(input_spec, test_metrics, server_optimizer, update_type='private'):
  """Build mirror descent iterative process."""

  def tff_model_fn():
    keras_model = models.create_recurrent_model(
        vocab_size=FLAGS.vocab_size,
        embedding_size=FLAGS.embedding_size,
        latent_size=FLAGS.latent_size,
        num_layers=FLAGS.num_layers,
        shared_embedding=FLAGS.shared_embedding,
        cell_type=FLAGS.lstm_cell)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return dp_fedavg.KerasModelWrapper(keras_model, input_spec, loss)

  noise_std = FLAGS.clip_norm * FLAGS.noise_multiplier / float(
      FLAGS.clients_per_round)
  server_optimizer_fn = functools.partial(
      _server_optimizer_fn,
      name=server_optimizer,
      learning_rate=FLAGS.server_lr,
      noise_std=0.0)
  client_optimizer_fn = functools.partial(
      _client_optimizer_fn,
      name=FLAGS.client_optimizer,
      learning_rate=FLAGS.client_lr)

  if FLAGS.experiment_type == 'mirror_descent_SO' or FLAGS.experiment_type == 'mirror_descent_SO_warmstart':
    iterative_process = mirror_descent.build_averaging_process(
        tff_model_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_optimizer_fn=client_optimizer_fn,
        update_type=update_type,
        dp_clip_norm=FLAGS.clip_norm,
        dp_noise_std=noise_std,
        private_lr=FLAGS.private_lr)
  elif FLAGS.experiment_type == 'mirror_descent_convex_SO' or FLAGS.experiment_type == 'mirror_descent_convex_SO_warmstart':
    iterative_process = mirror_descent_convex.build_averaging_process(
        tff_model_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_optimizer_fn=client_optimizer_fn,
        update_type=update_type,
        dp_clip_norm=FLAGS.clip_norm,
        dp_noise_std=noise_std,
        private_lr=FLAGS.private_lr,
        total_rounds=FLAGS.total_rounds) 
  model = tff_model_fn()

  def evaluate_fn(model_weights, dataset):
    model.from_weights(model_weights)
    metrics = dp_fedavg.keras_evaluate(model.keras_model, dataset, test_metrics)
    return collections.OrderedDict(
        (metric.name, metric.result().numpy()) for metric in metrics)

  server_state_update_fn = _build_server_state_epoch_update_fn(
      FLAGS.server_optimizer, tff_model_fn, server_optimizer_fn)
  return iterative_process, evaluate_fn, server_state_update_fn

# def _build_stackoverflow_SGD_process(input_spec, test_metrics):
#   """Build Keras opt checkpoint iterative process."""

#   def tff_model_fn():
#     keras_model = models.create_recurrent_model(
#         vocab_size=FLAGS.vocab_size,
#         embedding_size=FLAGS.embedding_size,
#         latent_size=FLAGS.latent_size,
#         num_layers=FLAGS.num_layers,
#         shared_embedding=FLAGS.shared_embedding,
#         cell_type=FLAGS.lstm_cell)
#     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#     return dp_fedavg.KerasModelWrapper(keras_model, input_spec, loss)

#   server_optimizer_fn = tf.keras.optimizers.SGD
#   client_optimizer_fn = tf.keras.optimizers.SGD
#   iterative_process = public.build_federated_averaging_process(
#         tff_model_fn,
#         dp_clip_norm=FLAGS.clip_norm,
#         server_optimizer_fn=server_optimizer_fn,
#         client_optimizer_fn=client_optimizer_fn,
#         server_learning_rate=FLAGS.server_lr,
#         server_momentum=FLAGS.server_momentum,
#         client_learning_rate=FLAGS.client_lr,
#         dp_noise_std=0.0)

#   model = tff_model_fn()

#   def evaluate_fn(model_weights, dataset):
#     model.from_weights(model_weights)
#     metrics = dp_fedavg.keras_evaluate(model.keras_model, dataset, test_metrics)
#     return collections.OrderedDict(
#         (metric.name, metric.result().numpy()) for metric in metrics)

#   server_state_update_fn = _build_server_state_epoch_update_fn(
#       FLAGS.server_optimizer, tff_model_fn, server_optimizer_fn)
#   return iterative_process, evaluate_fn, server_state_update_fn

def _build_tff_learning_model_and_process(input_spec, test_metrics,
                                          server_optimizer: str = 'sgd'):
  """Build `tff.learning` iterative process."""

  if FLAGS.server_optimizer != 'dpftrlm':
    raise ValueError(
        'When `use_tff_learning=True`, server optimizer must be `dpftrlm`'
        f'get {FLAGS.server_optimizer}.')

  if FLAGS.experiment_type != 'alternating':
    server_optimizer = FLAGS.server_optimizer

  def tff_model_fn():
    keras_model = models.create_recurrent_model(
        vocab_size=FLAGS.vocab_size,
        embedding_size=FLAGS.embedding_size,
        latent_size=FLAGS.latent_size,
        num_layers=FLAGS.num_layers,
        shared_embedding=FLAGS.shared_embedding,
        cell_type=FLAGS.lstm_cell)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return tff.learning.from_keras_model(
        keras_model=keras_model, input_spec=input_spec, loss=loss)

  def evaluate_fn(model_weights, dataset):
    keras_model = models.create_recurrent_model(
        vocab_size=FLAGS.vocab_size,
        embedding_size=FLAGS.embedding_size,
        latent_size=FLAGS.latent_size,
        num_layers=FLAGS.num_layers,
        shared_embedding=FLAGS.shared_embedding,
        cell_type=FLAGS.lstm_cell)
    model_weights.assign_weights_to(keras_model)
    metrics = dp_fedavg.keras_evaluate(keras_model, dataset, test_metrics)
    return collections.OrderedDict(
        (metric.name, metric.result().numpy()) for metric in metrics)

  client_optimizer_fn = functools.partial(
      _client_optimizer_fn,
      name=FLAGS.client_optimizer,
      learning_rate=FLAGS.client_lr)

  iterative_process = dp_fedavg.build_dpftrl_fedavg_process(
      tff_model_fn,
      client_optimizer_fn=client_optimizer_fn,
      server_learning_rate=FLAGS.server_lr,
      server_momentum=FLAGS.server_momentum,
      server_nesterov=False,
      clip_norm=FLAGS.clip_norm,
      noise_multiplier=FLAGS.noise_multiplier,
      report_goal=FLAGS.clients_per_round,
      noise_seed=None,
      use_experimental_simulation_loop=True)

  server_state_update_fn = None
  return iterative_process, evaluate_fn, server_state_update_fn


def train_and_eval_alternating():
  logging.info('Show FLAGS for debugging:')
  for f in HPARAM_FLAGS:
    logging.info('%s=%s', f, FLAGS[f].value)

  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in HPARAM_FLAGS
  ])

  if FLAGS.experiment_type == 'alternating_SO' or FLAGS.experiment_type == 'alternating_warmstart_SO':
     # Train on public SO
    train_dataset_computation_public, train_set_public, _, _ = _preprocess_data(
        'stackoverflow_public', FLAGS.vocab_size, FLAGS.num_oov_buckets,
        FLAGS.sequence_length, FLAGS.num_validation_examples,
        FLAGS.client_batch_size, FLAGS.client_epochs_per_round,
        FLAGS.max_elements_per_user)

    # Evaluate on StackOverflow
    train_dataset_computation_private, train_set_private, validation_set_private, test_set_private = _preprocess_data(
        'stackoverflow_private', FLAGS.vocab_size, FLAGS.num_oov_buckets,
        FLAGS.sequence_length, FLAGS.num_validation_examples,
        FLAGS.client_batch_size, FLAGS.client_epochs_per_round,
        FLAGS.max_elements_per_user)

  input_spec_private = train_dataset_computation_private.type_signature.result.element
  input_spec_public = train_dataset_computation_public.type_signature.result.element
  metrics = _get_metrics(FLAGS.vocab_size, FLAGS.num_oov_buckets)

  if FLAGS.use_tff_learning:
      iterative_process_private, evaluate_fn, server_state_update_fn = _build_tff_learning_model_and_process(
          input_spec_private, metrics, FLAGS.private_server_optimizer)
      iterative_process_public, _, _ = _build_tff_learning_model_and_process(
          input_spec_public, metrics, FLAGS.public_server_optimizer)
  else:
      iterative_process_private, evaluate_fn, server_state_update_fn = _build_custom_model_and_process(
          input_spec_private, metrics, FLAGS.private_server_optimizer)
      iterative_process_public, _, _ = _build_custom_model_and_process(
          input_spec_public, metrics, FLAGS.public_server_optimizer, FLAGS.public_clipping_strategy)

  iterative_process_private = tff.simulation.compose_dataset_computation_with_iterative_process(
        dataset_computation=train_dataset_computation_private,
        process=iterative_process_private)

  iterative_process_public = tff.simulation.compose_dataset_computation_with_iterative_process(
        dataset_computation=train_dataset_computation_public,
        process=iterative_process_public)

  if FLAGS.total_epochs is None:

      def client_dataset_ids_fn_private(round_num: int, epoch: int):
        return _sample_client_ids(FLAGS.private_round_size, train_set_private,
                                  round_num, epoch)

      logging.info('Sample clients for max %d rounds', FLAGS.total_rounds)
      total_epochs = 0

      def client_dataset_ids_fn_public(round_num: int, epoch: int):
        return _sample_client_ids(FLAGS.public_round_size, train_set_public,
                                  round_num, epoch)

      logging.info('Sample clients for max %d rounds', FLAGS.total_rounds)
      total_epochs = 0
  else:
      client_shuffer_private = training_loop.ClientIDShuffler(
          FLAGS.private_round_size, train_set_private)
      client_dataset_ids_fn_private = client_shuffer_private.sample_client_ids
      logging.info('Shuffle clients for max %d epochs and %d rounds',
                   FLAGS.total_epochs, FLAGS.total_rounds)
      total_epochs = FLAGS.total_epochs

      client_shuffer_public = training_loop.ClientIDShuffler(
          FLAGS.public_round_size, train_set_public)
      client_dataset_ids_fn_public = client_shuffer_public.sample_client_ids
      logging.info('Shuffle clients for max %d epochs and %d rounds',
                   FLAGS.total_epochs, FLAGS.total_rounds)
      total_epochs = FLAGS.total_epochs

  if 'warmstart' in FLAGS.experiment_type:
    alternating_loop.run(
          iterative_process_private,
          client_dataset_ids_fn_private,
          iterative_process_public,
          client_dataset_ids_fn_public,
          warmstart_file=FLAGS.warmstart_file,
          validation_fn=functools.partial(
              evaluate_fn, dataset=validation_set_private),
          total_epochs=total_epochs,
          total_rounds=FLAGS.total_rounds,
          experiment_name=FLAGS.experiment_name,
          train_eval_fn=None,
          test_fn=functools.partial(evaluate_fn, dataset=test_set_private),
          root_output_dir=FLAGS.root_output_dir,
          hparam_dict=hparam_dict,
          rounds_per_eval=FLAGS.rounds_per_eval,
          rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
          rounds_per_train_eval=2000,
          consecutive_private_rounds=FLAGS.consecutive_private_rounds,
          consecutive_public_rounds=FLAGS.consecutive_public_rounds,
          restart_optimizer=FLAGS.restart_optimizer,
          server_state_epoch_update_fn=server_state_update_fn)
  else:
    alternating_loop.run(
          iterative_process_private,
          client_dataset_ids_fn_private,
          iterative_process_public,
          client_dataset_ids_fn_public,
          validation_fn=functools.partial(
              evaluate_fn, dataset=validation_set_private),
          total_epochs=total_epochs,
          total_rounds=FLAGS.total_rounds,
          experiment_name=FLAGS.experiment_name,
          train_eval_fn=None,
          test_fn=functools.partial(evaluate_fn, dataset=test_set_private),
          root_output_dir=FLAGS.root_output_dir,
          hparam_dict=hparam_dict,
          rounds_per_eval=FLAGS.rounds_per_eval,
          rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
          rounds_per_train_eval=2000,
          consecutive_private_rounds=FLAGS.consecutive_private_rounds,
          consecutive_public_rounds=FLAGS.consecutive_public_rounds,
          restart_optimizer=FLAGS.restart_optimizer,
          server_state_epoch_update_fn=server_state_update_fn)

def train_and_eval_scaffold():
  logging.info('Show FLAGS for debugging:')
  for f in HPARAM_FLAGS:
    logging.info('%s=%s', f, FLAGS[f].value)

  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in HPARAM_FLAGS
  ])

  train_dataset_computation_private, train_dataset_computation_public, train_set_private, train_set_public, validation_set, test_set = _preprocess_scaffold_data(
        FLAGS.vocab_size, FLAGS.num_oov_buckets,
        FLAGS.sequence_length, FLAGS.num_validation_examples,
        FLAGS.client_batch_size, FLAGS.client_epochs_per_round,
        FLAGS.max_elements_per_user)

  input_spec_private = train_dataset_computation_private.type_signature.result.element
  input_spec_public = train_dataset_computation_public.type_signature.result.element
  metrics = _get_metrics(FLAGS.vocab_size, FLAGS.num_oov_buckets)

  iterative_process_private, evaluate_fn, server_state_update_fn, client_init_fn = _build_scaffold_model_and_process(
          input_spec_private, metrics, FLAGS.private_server_optimizer, 'private')
  iterative_process_public, _, _, _ = _build_scaffold_model_and_process(
          input_spec_public, metrics, FLAGS.public_server_optimizer, 'public')

  iterative_process_private = tff.simulation.compose_dataset_computation_with_iterative_process(
        dataset_computation=train_dataset_computation_private,
        process=iterative_process_private)

  iterative_process_public = tff.simulation.compose_dataset_computation_with_iterative_process(
        dataset_computation=train_dataset_computation_public,
        process=iterative_process_public)

  client_shuffer_private = training_loop.ClientIDShuffler(
          FLAGS.private_round_size, train_set_private)
  client_dataset_ids_fn_private = client_shuffer_private.sample_client_ids
  logging.info('Shuffle clients for max %d epochs and %d rounds',
                   FLAGS.total_epochs, FLAGS.total_rounds)

  client_shuffer_public = training_loop.ClientIDShuffler(
          FLAGS.public_round_size, train_set_public)
  client_dataset_ids_fn_public = client_shuffer_public.sample_client_ids
  logging.info('Shuffle clients for max %d epochs and %d rounds',
                   FLAGS.total_epochs, FLAGS.total_rounds)
  total_epochs = 0

  private_client_states = {
        client_id: None
        for client_id in train_set_private.client_ids

      }
  public_client_states = {
    client_id: None
    for client_id in train_set_public.client_ids
  }

  scaffold_loop.run(
        iterative_process_private,
        iterative_process_public,
        client_dataset_ids_fn_private,
        client_dataset_ids_fn_public,
        private_client_states,
        public_client_states,
        client_init_fn,
        validation_fn=functools.partial(
            evaluate_fn, dataset=validation_set),
        total_epochs=total_epochs,
        total_rounds=FLAGS.total_rounds,
        experiment_name=FLAGS.experiment_name,
        train_eval_fn=None,
        test_fn=functools.partial(evaluate_fn, dataset=test_set),
        root_output_dir=FLAGS.root_output_dir,
        hparam_dict=hparam_dict,
        rounds_per_eval=FLAGS.rounds_per_eval,
        rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
        rounds_per_train_eval=2000,
        server_state_epoch_update_fn=server_state_update_fn)

def train_and_eval_mime():
  logging.info('Show FLAGS for debugging:')
  for f in HPARAM_FLAGS:
    logging.info('%s=%s', f, FLAGS[f].value)

  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in HPARAM_FLAGS
  ])

  train_dataset_computation_private, train_dataset_computation_public, train_set_private, train_set_public, validation_set, test_set = _preprocess_mime_data(
        FLAGS.vocab_size, FLAGS.num_oov_buckets,
        FLAGS.sequence_length, FLAGS.num_validation_examples,
        FLAGS.client_batch_size, FLAGS.client_epochs_per_round,
        FLAGS.max_elements_per_user)

  client_ids_size = int(100)
  training_set_client_ids = random.sample(train_set_public.client_ids, client_ids_size)

  input_spec_private = train_dataset_computation_private.type_signature.result.element
  input_spec_public = train_dataset_computation_public.type_signature.result.element
  metrics = _get_metrics(FLAGS.vocab_size, FLAGS.num_oov_buckets)

  iterative_process_private, evaluate_fn, server_state_update_fn = _build_mime_model_and_process(
          input_spec_private, metrics, FLAGS.private_server_optimizer, 'private')
  iterative_process_public, _, _ = _build_mime_model_and_process(
          input_spec_public, metrics, FLAGS.public_server_optimizer, 'public')

  iterative_process_private = tff.simulation.compose_dataset_computation_with_iterative_process(
        dataset_computation=train_dataset_computation_private,
        process=iterative_process_private)

  iterative_process_public = tff.simulation.compose_dataset_computation_with_iterative_process(
        dataset_computation=train_dataset_computation_public,
        process=iterative_process_public)

  if FLAGS.total_epochs is None:

      def client_dataset_ids_fn_private(round_num: int, epoch: int):
        return _sample_client_ids(FLAGS.private_round_size, train_set_private,
                                  round_num, epoch)

      logging.info('Sample clients for max %d rounds', FLAGS.total_rounds)
      total_epochs = 0

      def client_dataset_ids_fn_public(round_num: int, epoch: int):
        if FLAGS.experiment_type == 'mime_SO' or FLAGS.experiment_type == 'mime_warmstart_SO':
          logging.info("Sampling from subset of public")
          return _sample_public_client_ids(FLAGS.clients_per_round, training_set_client_ids, round_num, epoch)
        else:
          return _sample_client_ids(FLAGS.public_round_size, train_set_public,
                                  round_num, epoch)

      logging.info('Sample clients for max %d rounds', FLAGS.total_rounds)
      total_epochs = 0
  else:
      client_shuffer_private = training_loop.ClientIDShuffler(
          FLAGS.private_round_size, train_set_private)
      client_dataset_ids_fn_private = client_shuffer_private.sample_client_ids
      logging.info('Shuffle clients for max %d epochs and %d rounds',
                   FLAGS.total_epochs, FLAGS.total_rounds)
      total_epochs = FLAGS.total_epochs

      client_shuffer_public = training_loop.ClientIDShuffler(
          FLAGS.public_round_size, train_set_public)
      client_dataset_ids_fn_public = client_shuffer_public.sample_client_ids
      logging.info('Shuffle clients for max %d epochs and %d rounds',
                   FLAGS.total_epochs, FLAGS.total_rounds)
      total_epochs = FLAGS.total_epochs

  if 'warmstart' in FLAGS.experiment_type:
    mime_loop.run(
        iterative_process_private,
        iterative_process_public,
        client_dataset_ids_fn_private,
        client_dataset_ids_fn_public,
        validation_fn=functools.partial(
            evaluate_fn, dataset=validation_set),
        total_epochs=total_epochs,
        total_rounds=FLAGS.total_rounds,
        experiment_name=FLAGS.experiment_name,
        warmstart_file=FLAGS.warmstart_file,
        train_eval_fn=None,
        test_fn=functools.partial(evaluate_fn, dataset=test_set),
        root_output_dir=FLAGS.root_output_dir,
        hparam_dict=hparam_dict,
        rounds_per_eval=FLAGS.rounds_per_eval,
        rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
        rounds_per_train_eval=2000,
        server_state_epoch_update_fn=server_state_update_fn)

  else:
    mime_loop.run(
        iterative_process_private,
        iterative_process_public,
        client_dataset_ids_fn_private,
        client_dataset_ids_fn_public,
        validation_fn=functools.partial(
            evaluate_fn, dataset=validation_set),
        total_epochs=total_epochs,
        total_rounds=FLAGS.total_rounds,
        experiment_name=FLAGS.experiment_name,
        train_eval_fn=None,
        test_fn=functools.partial(evaluate_fn, dataset=test_set),
        root_output_dir=FLAGS.root_output_dir,
        hparam_dict=hparam_dict,
        rounds_per_eval=FLAGS.rounds_per_eval,
        rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
        rounds_per_train_eval=2000,
        server_state_epoch_update_fn=server_state_update_fn)


def train_and_eval_mimelite():
  logging.info('Show FLAGS for debugging:')
  for f in HPARAM_FLAGS:
    logging.info('%s=%s', f, FLAGS[f].value)

  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in HPARAM_FLAGS
  ])

  train_dataset_computation_private, train_dataset_computation_public, train_set_private, train_set_public, validation_set, test_set = _preprocess_mimelite_data(
        FLAGS.vocab_size, FLAGS.num_oov_buckets,
        FLAGS.sequence_length, FLAGS.num_validation_examples,
        FLAGS.client_batch_size, FLAGS.client_epochs_per_round,
        FLAGS.max_elements_per_user)

  client_ids_size = int(100)
  training_set_client_ids = random.sample(train_set_public.client_ids, client_ids_size)

  input_spec_private = train_dataset_computation_private.type_signature.result.element
  input_spec_public = train_dataset_computation_public.type_signature.result.element
  metrics = _get_metrics(FLAGS.vocab_size, FLAGS.num_oov_buckets)

  iterative_process_private, evaluate_fn, server_state_update_fn = _build_mimelite_model_and_process(
          input_spec_private, metrics, FLAGS.private_server_optimizer, 'private')
  iterative_process_public, _, _ = _build_mimelite_model_and_process(
          input_spec_public, metrics, FLAGS.public_server_optimizer, 'public')

  iterative_process_private = tff.simulation.compose_dataset_computation_with_iterative_process(
        dataset_computation=train_dataset_computation_private,
        process=iterative_process_private)

  iterative_process_public = tff.simulation.compose_dataset_computation_with_iterative_process(
        dataset_computation=train_dataset_computation_public,
        process=iterative_process_public)
  if FLAGS.total_epochs is None:

      def client_dataset_ids_fn_private(round_num: int, epoch: int):
        return _sample_client_ids(FLAGS.private_round_size, train_set_private,
                                  round_num, epoch)

      logging.info('Sample clients for max %d rounds', FLAGS.total_rounds)
      total_epochs = 0

      def client_dataset_ids_fn_public(round_num: int, epoch: int):
        if FLAGS.experiment_type == 'mimelite_SO' or FLAGS.experiment_type == 'mimelite_warmstart_SO':
          logging.info("Sampling from subset of public")
          return _sample_public_client_ids(FLAGS.clients_per_round, training_set_client_ids, round_num, epoch)
        else:
          return _sample_client_ids(FLAGS.public_round_size, train_set_public,
                                  round_num, epoch)

      logging.info('Sample clients for max %d rounds', FLAGS.total_rounds)
      total_epochs = 0
  else:
      client_shuffer_private = training_loop.ClientIDShuffler(
          FLAGS.private_round_size, train_set_private)
      client_dataset_ids_fn_private = client_shuffer_private.sample_client_ids
      logging.info('Shuffle clients for max %d epochs and %d rounds',
                   FLAGS.total_epochs, FLAGS.total_rounds)
      total_epochs = FLAGS.total_epochs

      client_shuffer_public = training_loop.ClientIDShuffler(
          FLAGS.public_round_size, train_set_public)
      client_dataset_ids_fn_public = client_shuffer_public.sample_client_ids
      logging.info('Shuffle clients for max %d epochs and %d rounds',
                   FLAGS.total_epochs, FLAGS.total_rounds)
      total_epochs = FLAGS.total_epochs

  if 'warmstart' in FLAGS.experiment_type:
    mimelite_loop.run(
          iterative_process_private,
          iterative_process_public,
          client_dataset_ids_fn_private,
          client_dataset_ids_fn_public,
          validation_fn=functools.partial(
              evaluate_fn, dataset=validation_set),
          total_epochs=total_epochs,
          total_rounds=FLAGS.total_rounds,
          experiment_name=FLAGS.experiment_name,
          warmstart_file=FLAGS.warmstart_file,
          train_eval_fn=None,
          test_fn=functools.partial(evaluate_fn, dataset=test_set),
          root_output_dir=FLAGS.root_output_dir,
          hparam_dict=hparam_dict,
          rounds_per_eval=FLAGS.rounds_per_eval,
          rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
          rounds_per_train_eval=2000,
          server_state_epoch_update_fn=server_state_update_fn)
  else:
   mimelite_loop.run(
          iterative_process_private,
          iterative_process_public,
          client_dataset_ids_fn_private,
          client_dataset_ids_fn_public,
          validation_fn=functools.partial(
              evaluate_fn, dataset=validation_set),
          total_epochs=total_epochs,
          total_rounds=FLAGS.total_rounds,
          experiment_name=FLAGS.experiment_name,
          train_eval_fn=None,
          test_fn=functools.partial(evaluate_fn, dataset=test_set),
          root_output_dir=FLAGS.root_output_dir,
          hparam_dict=hparam_dict,
          rounds_per_eval=FLAGS.rounds_per_eval,
          rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
          rounds_per_train_eval=2000,
          server_state_epoch_update_fn=server_state_update_fn)


def train_and_eval():
  """Train and evaluate StackOver NWP task."""
  logging.info('Show FLAGS for debugging:')
  for f in HPARAM_FLAGS:
    logging.info('%s=%s', f, FLAGS[f].value)

  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in HPARAM_FLAGS
  ])

  if FLAGS.experiment_type == 'private':
    # Evaluate on StackOverflow
    train_dataset_computation, train_set, validation_set, test_set = (
        _preprocess_data('stackoverflow_private', FLAGS.vocab_size,
                         FLAGS.num_oov_buckets, FLAGS.sequence_length,
                         FLAGS.num_validation_examples, FLAGS.client_batch_size,
                         FLAGS.client_epochs_per_round,
                         FLAGS.max_elements_per_user))

  elif FLAGS.experiment_type == 'public_SO' or FLAGS.experiment_type == 'stackoverflow_SGD':
     # Evaluate on StackOverflow
    _, train_set_private, _, _ = (
        _preprocess_data('stackoverflow_private', FLAGS.vocab_size,
                         FLAGS.num_oov_buckets, FLAGS.sequence_length,
                         FLAGS.num_validation_examples, FLAGS.client_batch_size,
                         FLAGS.client_epochs_per_round,
                         FLAGS.max_elements_per_user))

    train_dataset_computation, train_set, validation_set, test_set = (
        _preprocess_data('stackoverflow_public', FLAGS.vocab_size,
                         FLAGS.num_oov_buckets, FLAGS.sequence_length,
                         FLAGS.num_validation_examples, FLAGS.client_batch_size,
                         FLAGS.client_epochs_per_round,
                         FLAGS.max_elements_per_user))

    client_ids_size = int(100)
    training_set_client_ids = random.sample(train_set.client_ids, client_ids_size)

  elif FLAGS.experiment_type == 'warmstart':
    # Evaluate on StackOverflow
    train_dataset_computation, train_set, validation_set, test_set = (
        _preprocess_data('stackoverflow_private', FLAGS.vocab_size,
                         FLAGS.num_oov_buckets, FLAGS.sequence_length,
                         FLAGS.num_validation_examples, FLAGS.client_batch_size,
                         FLAGS.client_epochs_per_round,
                         FLAGS.max_elements_per_user))
    input_spec = train_dataset_computation.type_signature.result.element
    metrics = _get_metrics(FLAGS.vocab_size, FLAGS.num_oov_buckets)

    if FLAGS.use_tff_learning:
     iterative_process, evaluate_fn, server_state_update_fn = _build_tff_learning_model_and_process(
        input_spec, metrics)
    else:
     iterative_process, evaluate_fn, server_state_update_fn = _build_custom_model_and_process(
        input_spec, metrics)

    iterative_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      dataset_computation=train_dataset_computation, process=iterative_process)

    if FLAGS.total_epochs is None:

     def client_dataset_ids_fn(round_num: int, epoch: int):
       return _sample_client_ids(FLAGS.clients_per_round, train_set, round_num,
                                epoch)

     logging.info('Sample clients for max %d rounds', FLAGS.total_rounds)
     total_epochs = 0
    else:
     client_shuffer = training_loop.ClientIDShuffler(FLAGS.clients_per_round,
                                                    train_set)
     client_dataset_ids_fn = client_shuffer.sample_client_ids
     logging.info('Shuffle clients for max %d epochs and %d rounds',
                 FLAGS.total_epochs, FLAGS.total_rounds)
     total_epochs = FLAGS.total_epochs

    warmstart_loop.run(
      iterative_process,
      client_dataset_ids_fn,
      warmstart_file=FLAGS.warmstart_file,
      validation_fn=functools.partial(evaluate_fn, dataset=validation_set),
      total_epochs=total_epochs,
      total_rounds=FLAGS.total_rounds,
      experiment_name=FLAGS.experiment_name,
      train_eval_fn=None,
      test_fn=functools.partial(evaluate_fn, dataset=test_set),
      root_output_dir=FLAGS.root_output_dir,
      hparam_dict=hparam_dict,
      rounds_per_eval=FLAGS.rounds_per_eval,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
      rounds_per_train_eval=2000,
      server_state_epoch_update_fn=server_state_update_fn)
    return

  else:
    raise ValueError('Experiment type is not supported %s'.format(
        FLAGS.experiment_type))

  input_spec = train_dataset_computation.type_signature.result.element
  metrics = _get_metrics(FLAGS.vocab_size, FLAGS.num_oov_buckets)

  if FLAGS.use_tff_learning:
    iterative_process, evaluate_fn, server_state_update_fn = _build_tff_learning_model_and_process(
        input_spec, metrics)
  elif FLAGS.experiment_type == 'stackoverflow_SGD':
    iterative_process, evaluate_fn, server_state_update_fn = _build_stackoverflow_SGD_process(
        input_spec, metrics)
  else:
    iterative_process, evaluate_fn, server_state_update_fn = _build_custom_model_and_process(
        input_spec, metrics)
  iterative_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      dataset_computation=train_dataset_computation, process=iterative_process)

  if FLAGS.total_epochs is None:

    def client_dataset_ids_fn(round_num: int, epoch: int):
      if FLAGS.experiment_type == 'public_SO' or FLAGS.experiment_type == 'stackoverflow_SGD':
        logging.info("Sampling from subset of public")
        return _sample_public_client_ids(FLAGS.clients_per_round, training_set_client_ids, round_num, epoch)
      else:
        return _sample_client_ids(FLAGS.clients_per_round, train_set, round_num,
                                epoch)

    logging.info('Sample clients for max %d rounds', FLAGS.total_rounds)
    total_epochs = 0
  else:
    client_shuffer = training_loop.ClientIDShuffler(FLAGS.clients_per_round,
                                                    train_set)
    client_dataset_ids_fn = client_shuffer.sample_client_ids
    logging.info('Shuffle clients for max %d epochs and %d rounds',
                 FLAGS.total_epochs, FLAGS.total_rounds)
    total_epochs = FLAGS.total_epochs

  if FLAGS.experiment_type != 'stackoverflow_SGD':
    training_loop.run(
        iterative_process,
        client_dataset_ids_fn,
        validation_fn=functools.partial(evaluate_fn, dataset=validation_set),
        total_epochs=total_epochs,
        total_rounds=FLAGS.total_rounds,
        experiment_name=FLAGS.experiment_name,
        train_eval_fn=None,
        test_fn=functools.partial(evaluate_fn, dataset=test_set),
        root_output_dir=FLAGS.root_output_dir,
        hparam_dict=hparam_dict,
        rounds_per_eval=FLAGS.rounds_per_eval,
        rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
        rounds_per_train_eval=2000,
        server_state_epoch_update_fn=server_state_update_fn)

def train_and_eval_mirror_descent():
  logging.info('Show FLAGS for debugging:')
  for f in HPARAM_FLAGS:
    logging.info('%s=%s', f, FLAGS[f].value)

  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in HPARAM_FLAGS
  ])

  if FLAGS.experiment_type == 'mirror_descent_SO' or FLAGS.experiment_type == 'mirror_descent_warmstart_SO' or FLAGS.experiment_type == 'mirror_descent_convex_SO' or FLAGS.experiment_type == 'mirror_descent_convex_warmstart_SO':
     # Train on public SO
    train_dataset_computation_public, train_set_public, _, _ = _preprocess_data(
        'stackoverflow_public', FLAGS.vocab_size, FLAGS.num_oov_buckets,
        FLAGS.sequence_length, FLAGS.num_validation_examples,
        FLAGS.client_batch_size, FLAGS.client_epochs_per_round,
        FLAGS.max_elements_per_user)

    # Evaluate on StackOverflow
    train_dataset_computation_private, train_set_private, validation_set_private, test_set_private = _preprocess_data(
        'stackoverflow_private', FLAGS.vocab_size, FLAGS.num_oov_buckets,
        FLAGS.sequence_length, FLAGS.num_validation_examples,
        FLAGS.client_batch_size, FLAGS.client_epochs_per_round,
        FLAGS.max_elements_per_user)

    client_ids_size = int(100)
    training_set_client_ids = random.sample(train_set_public.client_ids, client_ids_size)

  input_spec_private = train_dataset_computation_private.type_signature.result.element
  input_spec_public = train_dataset_computation_public.type_signature.result.element
  metrics = _get_metrics(FLAGS.vocab_size, FLAGS.num_oov_buckets)

  if FLAGS.use_tff_learning:
      iterative_process_private, evaluate_fn, server_state_update_fn = _build_tff_learning_model_and_process(
          input_spec_private, metrics, FLAGS.private_server_optimizer)
      iterative_process_public, _, _ = _build_tff_learning_model_and_process(
          input_spec_public, metrics, FLAGS.public_server_optimizer)
  else:
      iterative_process_private, evaluate_fn, server_state_update_fn = _build_mirror_descent_model_and_process(
          input_spec_private, metrics, FLAGS.private_server_optimizer, 'private')
      iterative_process_public, _, _ = _build_mirror_descent_model_and_process(
          input_spec_public, metrics, FLAGS.public_server_optimizer, 'public')
      iterative_process_public_old, _, _ = _build_mirror_descent_model_and_process(
          input_spec_public, metrics, FLAGS.public_server_optimizer, 'public_old')


  iterative_process_private = tff.simulation.compose_dataset_computation_with_iterative_process(
        dataset_computation=train_dataset_computation_private,
        process=iterative_process_private)

  iterative_process_public = tff.simulation.compose_dataset_computation_with_iterative_process(
        dataset_computation=train_dataset_computation_public,
        process=iterative_process_public)

  iterative_process_public_old = tff.simulation.compose_dataset_computation_with_iterative_process(
        dataset_computation=train_dataset_computation_public,
        process=iterative_process_public_old)

  if FLAGS.total_epochs is None:

      def client_dataset_ids_fn_private(round_num: int, epoch: int):
        return _sample_client_ids(FLAGS.private_round_size, train_set_private,
                                  round_num, epoch)

      logging.info('Sample clients for max %d rounds', FLAGS.total_rounds)
      total_epochs = 0

      def client_dataset_ids_fn_public(round_num: int, epoch: int):
        if FLAGS.experiment_type == 'mirror_descent_SO' or FLAGS.experiment_type == 'mirror_descent_warmstart_SO' or FLAGS.experiment_type == 'mirror_descent_convex_SO' or FLAGS.experiment_type == 'mirror_descent_convex_warmstart_SO':
          logging.info("Sampling from subset of public")
          return _sample_public_client_ids(FLAGS.clients_per_round, training_set_client_ids, round_num, epoch)
        else:
          return _sample_client_ids(FLAGS.public_round_size, train_set_public,
                                  round_num, epoch)

      logging.info('Sample clients for max %d rounds', FLAGS.total_rounds)
      total_epochs = 0
  else:
      client_shuffer_private = training_loop.ClientIDShuffler(
          FLAGS.private_round_size, train_set_private)
      client_dataset_ids_fn_private = client_shuffer_private.sample_client_ids
      logging.info('Shuffle clients for max %d epochs and %d rounds',
                   FLAGS.total_epochs, FLAGS.total_rounds)
      total_epochs = FLAGS.total_epochs

      client_shuffer_public = training_loop.ClientIDShuffler(
          FLAGS.public_round_size, train_set_public)
      client_dataset_ids_fn_public = client_shuffer_public.sample_client_ids
      logging.info('Shuffle clients for max %d epochs and %d rounds',
                   FLAGS.total_epochs, FLAGS.total_rounds)
      total_epochs = FLAGS.total_epochs

  if 'warmstart' in FLAGS.experiment_type:
    mirror_descent_loop.run(
          iterative_process_private,
          client_dataset_ids_fn_private,
          iterative_process_public,
          client_dataset_ids_fn_public,
          iterative_process_public_old,
          warmstart_file=FLAGS.warmstart_file,
          validation_fn=functools.partial(
              evaluate_fn, dataset=validation_set_private),
          total_epochs=total_epochs,
          total_rounds=FLAGS.total_rounds,
          experiment_name=FLAGS.experiment_name,
          train_eval_fn=None,
          test_fn=functools.partial(evaluate_fn, dataset=test_set_private),
          root_output_dir=FLAGS.root_output_dir,
          hparam_dict=hparam_dict,
          rounds_per_eval=FLAGS.rounds_per_eval,
          rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
          rounds_per_train_eval=2000,
          update_private_gradient_frequency=FLAGS.update_private_gradient_frequency,
          restart_optimizer=FLAGS.restart_optimizer,
          server_state_epoch_update_fn=server_state_update_fn)
  else:
    mirror_descent_loop.run(
          iterative_process_private,
          client_dataset_ids_fn_private,
          iterative_process_public,
          client_dataset_ids_fn_public,
          iterative_process_public_old,
          validation_fn=functools.partial(
              evaluate_fn, dataset=validation_set_private),
          total_epochs=total_epochs,
          total_rounds=FLAGS.total_rounds,
          experiment_name=FLAGS.experiment_name,
          train_eval_fn=None,
          test_fn=functools.partial(evaluate_fn, dataset=test_set_private),
          root_output_dir=FLAGS.root_output_dir,
          hparam_dict=hparam_dict,
          rounds_per_eval=FLAGS.rounds_per_eval,
          rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
          rounds_per_train_eval=2000,
          update_private_gradient_frequency=FLAGS.update_private_gradient_frequency,
          restart_optimizer=FLAGS.restart_optimizer,
          server_state_epoch_update_fn=server_state_update_fn)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  # Multi-GPU configuration
  print(argv)
  client_devices = tf.config.list_logical_devices('GPU')
  server_device = tf.config.list_logical_devices('CPU')[0]
  tff.backends.native.set_local_execution_context(
      max_fanout=2 * FLAGS.clients_per_round,
      server_tf_device=server_device,
      client_tf_devices=client_devices,
      clients_per_thread=FLAGS.clients_per_thread)
  print(FLAGS.experiment_type)
  if 'alternating' == FLAGS.experiment_type or 'alternating_SO' == FLAGS.experiment_type or 'alternating_warmstart' == FLAGS.experiment_type or 'alternating_warmstart_SO' == FLAGS.experiment_type:
    train_and_eval_alternating()
  elif 'scaffold' == FLAGS.experiment_type or 'scaffold_SO' == FLAGS.experiment_type:
    train_and_eval_scaffold()
  elif 'mime' == FLAGS.experiment_type or 'mime_SO' == FLAGS.experiment_type or 'mime_warmstart' == FLAGS.experiment_type or 'mime_warmstart_SO' == FLAGS.experiment_type:
    train_and_eval_mime()
  elif 'mimelite' == FLAGS.experiment_type or 'mimelite_SO' == FLAGS.experiment_type or 'mimelite_warmstart' == FLAGS.experiment_type or 'mimelite_warmstart_SO' == FLAGS.experiment_type:
    train_and_eval_mimelite()
  elif 'mirror_descent' == FLAGS.experiment_type or 'mirror_descent_SO' == FLAGS.experiment_type or 'mirror_descent_warmstart' == FLAGS.experiment_type or 'mirror_descent_warmstart_SO' == FLAGS.experiment_type:
    train_and_eval_mirror_descent()
  elif 'mirror_descent_convex' == FLAGS.experiment_type or 'mirror_descent_convex_SO' == FLAGS.experiment_type or 'mirror_descent_convex_warmstart' == FLAGS.experiment_type or 'mirror_descent_convex_warmstart_SO' == FLAGS.experiment_type:
    train_and_eval_mirror_descent()
  else:
    train_and_eval()

if __name__ == '__main__':
  app.run(main)
