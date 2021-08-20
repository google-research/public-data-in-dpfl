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
"""Implementation of average word perplexity metric.
"""

import tensorflow as tf


def per_position_loss(labels, logits, padding_variable=0, name='per_position_loss'):
  with tf.name_scope(name):
    labels = tf.convert_to_tensor(labels)
    logits = tf.convert_to_tensor(logits)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    mask = tf.cast(labels != padding_variable, dtype=loss.dtype)
    return loss * mask


class Perplexity(tf.keras.metrics.Metric):
  """Average word perplexity eval metric."""

  def __init__(self, padding_variable, name='perplexity', **kwargs):
    super(Perplexity, self).__init__(name=name, **kwargs)
    self._num_words = self.add_weight(name='num_words', initializer='zeros')
    self._log_prob = self.add_weight(name='log_prob', initializer='zeros')
    self._padding_variable = padding_variable

  def update_state(self, y_true, y_pred):
    num_words = tf.math.count_nonzero(y_true)
    log_prob = -tf.reduce_sum(per_position_loss(labels=y_true, logits=y_pred, padding_variable=self._padding_variable))
    self._num_words.assign_add(tf.cast(num_words, self._num_words.dtype))
    self._log_prob.assign_add(log_prob)

  def result(self):
    return tf.exp(-self._log_prob / self._num_words)
