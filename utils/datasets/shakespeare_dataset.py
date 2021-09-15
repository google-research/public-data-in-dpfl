# Copyright 2019, Google LLC.
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
"""Libraries to prepare Shakespeare datasets for CharRNN experiments."""

from typing import Callable, Tuple

import tensorflow as tf
import tensorflow_federated as tff

SEQUENCE_LENGTH = 80  # from McMahan et al AISTATS 2017
# Vocabulary re-used from the Federated Learning for Text Generation tutorial.
# https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
CHAR_VOCAB = list(
    'dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r'
)
EVAL_BATCH_SIZE = 10


def get_special_tokens() -> Tuple[int, int, int, int]:
  """Gets tokens dataset preprocessing code will add to Shakespeare.

  Returns:
    A tuple of the four special characters, (pad, oov, bos, eos).

  """
  vocab_size = len(CHAR_VOCAB)
  pad = 0
  oov = vocab_size + 1
  bos = vocab_size + 2
  eos = vocab_size + 3
  return pad, oov, bos, eos


def _build_tokenize_fn(split_length: int = SEQUENCE_LENGTH + 1):
  """Create a tf.function that converts a Shakespeare example to character ids.

  The function converts each example to its corresponding character ids. It then
  pads the sequence until its length is a multiple of split_length.

  Args:
    split_length: An integer used to determine the padding length for a given
      snippet. The tf.function pads until the sequence is of length divisible by
      split_length. This function is intended to be used in combination with
      something such as batching, in order to create token sequences of length
      split_length.

  Returns:
    A `tf.function`.
  """
  _, _, bos, eos = get_special_tokens()

  ids = tf.range(len(CHAR_VOCAB), dtype=tf.int64)
  lookup_table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(CHAR_VOCAB, ids), num_oov_buckets=1)

  def to_tokens_and_pad(example: tf.Tensor) -> tf.Tensor:
    """Convert a Shakespeare example to a int64 tensor of token ids, and pad."""
    chars = tf.strings.bytes_split(example['snippets'])
    tokens = lookup_table.lookup(keys=chars) + 1  # Reserve 0 for pad.
    tokens = tf.concat([[bos], tokens, [eos]], 0)
    pad_length = (-tf.shape(tokens)[0]) % split_length
    return tf.concat([tokens, tf.zeros(pad_length, dtype=tf.int64)], 0)

  return to_tokens_and_pad


def _split_target(sequence_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Split a N + 1 sequence into shifted-by-1 sequences for input and output."""
  input_text = tf.map_fn(lambda x: x[:-1], sequence_batch)
  target_text = tf.map_fn(lambda x: x[1:], sequence_batch)
  return (input_text, target_text)


def create_preprocess_fn(
    num_epochs: int,
    batch_size: int,
    shuffle_buffer_size: int = 50,
    sequence_length: int = SEQUENCE_LENGTH,
    num_parallel_calls: int = tf.data.experimental.AUTOTUNE
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """Creates a preprocessing function for Shakespeare client datasets.

  This function maps a dataset of string snippets to a dataset of input/output
  character ID sequences. This is done by first repeating the dataset and
  shuffling (according to `num_epochs` and `shuffle_buffer_size`), mapping
  the the string sequences to tokens, and packing them into input/output
  sequences of length `sequence_length`.

  Args:
    num_epochs: An integer representing the number of epochs to repeat the
      client datasets.
    batch_size: An integer representing the batch size on clients.
    shuffle_buffer_size: An integer representing the shuffle buffer size on
      clients. If set to a number <= 1, no shuffling occurs.
    sequence_length: the length of each example in the batch.
    num_parallel_calls: An integer representing the number of parallel calls
      used when performing `tf.data.Dataset.map`.

  Returns:
    A callable performing the preprocessing described above.
  """
  if num_epochs < 1:
    raise ValueError('num_epochs must be a positive integer.')
  if sequence_length < 1:
    raise ValueError('sequence_length must be a positive integer.')
  if shuffle_buffer_size <= 1:
    shuffle_buffer_size = 1

  def preprocess_fn(dataset):
    to_tokens = _build_tokenize_fn(split_length=sequence_length + 1)
    return (
        dataset.shuffle(shuffle_buffer_size).repeat(num_epochs)
        # Convert snippets to int64 tokens and pad.
        .map(to_tokens, num_parallel_calls=num_parallel_calls)
        # Separate into individual tokens
        .unbatch()
        # Join into sequences of the desired length. The previous call of
        # map(to_ids,...) ensures that the collection of tokens has length
        # divisible by sequence_length + 1, so no batch dropping is expected.
        .batch(sequence_length + 1, drop_remainder=True)
        # Batch sequences together for mini-batching purposes.
        .batch(batch_size)
        # Convert batches into training examples.
        .map(_split_target, num_parallel_calls=num_parallel_calls))

  return preprocess_fn


def get_federated_datasets(
    train_client_batch_size: int = 4,
    test_client_batch_size: int = 100,
    train_client_epochs_per_round: int = 1,
    test_client_epochs_per_round: int = 1,
    train_shuffle_buffer_size: int = 50,
    test_shuffle_buffer_size: int = 1,
    sequence_length: int = SEQUENCE_LENGTH
) -> Tuple[tff.simulation.datasets.ClientData,
           tff.simulation.datasets.ClientData]:
  """Loads and preprocesses federated Shakespeare datasets.

  Args:
    train_client_batch_size: The batch size for all train clients.
    test_client_batch_size: The batch size for all test clients.
    train_client_epochs_per_round: The number of epochs each train client should
      iterate over their local dataset, via `tf.data.Dataset.repeat`. Must be
      set to a positive integer.
    test_client_epochs_per_round: The number of epochs each test client should
      iterate over their local dataset, via `tf.data.Dataset.repeat`. Must be
      set to a positive integer.
    train_shuffle_buffer_size: An integer representing the shuffle buffer size
      (as in `tf.data.Dataset.shuffle`) for each train client's dataset. By
      default, this is set to the largest dataset size among all clients. If set
      to some integer less than or equal to 1, no shuffling occurs.
    test_shuffle_buffer_size: An integer representing the shuffle buffer size
      (as in `tf.data.Dataset.shuffle`) for each test client's dataset. If set
      to some integer less than or equal to 1, no shuffling occurs.
    sequence_length: The resulting length of input/output sequences in the
      client datasets.

  Returns:
    A tuple (shakespeare_train, shakespeare_test) of
    `tff.simulation.datasets.ClientData` instances representing the federated
    training and test datasets.
  """
  if train_client_epochs_per_round < 1:
    raise ValueError(
        'train_client_epochs_per_round must be a positive integer.')
  if test_client_epochs_per_round < 0:
    raise ValueError('test_client_epochs_per_round must be a positive integer.')
  if train_shuffle_buffer_size <= 1:
    train_shuffle_buffer_size = 1
  if test_shuffle_buffer_size <= 1:
    test_shuffle_buffer_size = 1

  shakespeare_train, shakespeare_test = (
      tff.simulation.datasets.shakespeare.load_data())

  train_preprocess_fn = create_preprocess_fn(
      num_epochs=train_client_epochs_per_round,
      batch_size=train_client_batch_size,
      shuffle_buffer_size=train_shuffle_buffer_size,
      sequence_length=sequence_length)

  test_preprocess_fn = create_preprocess_fn(
      num_epochs=test_client_epochs_per_round,
      batch_size=test_client_batch_size,
      shuffle_buffer_size=test_shuffle_buffer_size,
      sequence_length=sequence_length)

  shakespeare_train = shakespeare_train.preprocess(train_preprocess_fn)
  shakespeare_test = shakespeare_test.preprocess(test_preprocess_fn)
  return shakespeare_train, shakespeare_test


def get_centralized_datasets(
    train_batch_size: int = 20,
    test_batch_size: int = 100,
    train_shuffle_buffer_size: int = 1000,
    test_shuffle_buffer_size: int = 1,
    sequence_length: int = SEQUENCE_LENGTH
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Loads and preprocesses centralized Shakespeare datasets.

  Args:
    train_batch_size: The batch size for the training dataset.
    test_batch_size: The batch size for the test dataset.
    train_shuffle_buffer_size: An integer specifying the buffer size used to
      shuffle the train dataset via `tf.data.Dataset.shuffle`. If set to an
      integer less than or equal to 1, no shuffling occurs.
    test_shuffle_buffer_size: An integer specifying the buffer size used to
      shuffle the test dataset via `tf.data.Dataset.shuffle`. If set to an
      integer less than or equal to 1, no shuffling occurs.
    sequence_length: The number of characters in the input and output sequence
      of each example.

  Returns:
    A tuple (shakespeare_train, shakespeare_test) of `tf.data.Dataset` instances
    representing the centralized training and test datasets.
  """
  if train_shuffle_buffer_size <= 1:
    train_shuffle_buffer_size = 1
  if test_shuffle_buffer_size <= 1:
    test_shuffle_buffer_size = 1

  shakespeare_train, shakespeare_test = (
      tff.simulation.datasets.shakespeare.load_data())

  shakespeare_train = shakespeare_train.create_tf_dataset_from_all_clients()
  shakespeare_test = shakespeare_test.create_tf_dataset_from_all_clients()

  train_preprocess_fn = create_preprocess_fn(
      num_epochs=1,
      batch_size=train_batch_size,
      shuffle_buffer_size=train_shuffle_buffer_size,
      sequence_length=sequence_length)

  test_preprocess_fn = create_preprocess_fn(
      num_epochs=1,
      batch_size=test_batch_size,
      shuffle_buffer_size=test_shuffle_buffer_size,
      sequence_length=sequence_length)

  shakespeare_train = train_preprocess_fn(shakespeare_train)
  shakespeare_test = test_preprocess_fn(shakespeare_test)
  return shakespeare_train, shakespeare_test
