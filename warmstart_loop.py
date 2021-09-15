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
"""DP-FedAvg + warmstarted model with private updates training loop."""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from absl import logging
import tensorflow_federated as tff

from dp_ftrl import dp_fedavg
from dp_ftrl import training_loop


def run(
    iterative_process: tff.templates.IterativeProcess,
    client_datasets_fn: Callable[[int, int], Tuple[List, int]],  # pylint: disable=g-bare-generic
    warmstart_file: str,
    validation_fn: Callable[[Any], Dict[str, float]],
    total_epochs: int,
    total_rounds: int,
    experiment_name: str,
    train_eval_fn: Optional[Callable[[Any], Dict[str, float]]] = None,
    test_fn: Optional[Callable[[Any], Dict[str, float]]] = None,
    root_output_dir: Optional[str] = '/tmp/fed_opt',
    hparam_dict: Optional[Dict[str, Any]] = None,
    rounds_per_eval: Optional[int] = 1,
    rounds_per_checkpoint: Optional[int] = 50,
    rounds_per_train_eval: Optional[int] = 100,
    server_state_epoch_update_fn: Optional[Callable[
        [dp_fedavg.ServerState], dp_fedavg.ServerState]] = None):
  """Runs federated training for a given `tff.templates.IterativeProcess`.

  We assume that the iterative process has the following functional type
  signatures:

    *   `initialize`: `( -> S@SERVER)` where `S` represents the server state.
    *   `next`: `<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>` where `S`
        represents the server state, `{B*}` represents the client datasets,
        and `T` represents a python `Mapping` object.

  Args:
    iterative_process_private: A private `tff.templates.IterativeProcess`
      instance to run.
    client_datasets_fn_private: Function accepts integer arguments (the round
      number and the epoch) and returns a tuple of a list of private client
      datasets to use as data data for that round, and the updated epoch index.
    warmstart_file: File to checkpoint to start training from
      (typically a warmstarted model).
    validation_fn: A callable accepting the `model` attribute of the iterative
      process state and returning a dict of evaluation metrics. Used to compute
      validation metrics throughout the training process.
    total_epochs: Nubmer of total epochs if using `ClientIDShuffler` to shuffle
      clients. Use 0 when sampling clients and control by `total_rounds`.
    total_rounds: The number of federated training rounds to perform. If
      `ClientIDShuffler` is used for `client_datasets_fn`, the total rounds will
      take the minimum of `total_rounds` and rounds_per_epoch*`total_epochs`.
    experiment_name: The name of the experiment being run. This will be appended
      to the `root_output_dir` for purposes of writing outputs.
    train_eval_fn: An optional callable accepting the `model` attribute of the
      iterative process state and returning a dict of evaluation metrics. Used
      to compute training metrics over the entire training dataset throughout
      the course of the iterative process. If set to `None`, no such evaluation
      is done.
    test_fn: An optional callable accepting the `model` attribute of the
      iterative process state and returning a dict of test metrics. Used to
      compute test metrics at the end of the training process.
    root_output_dir: The name of the root output directory for writing
      experiment outputs.
    hparam_dict: An optional dictionary specifying hyperparameters of the
      experiment. If provided, the hyperparameters will be written to CSV.
    rounds_per_eval: How often to compute validation metrics.
    rounds_per_checkpoint: How often to checkpoint the iterative process state.
      If you expect the job to restart frequently, this should be small. If no
      interruptions are expected, this can be made larger.
    rounds_per_train_eval: How often to compute metrics over the entire training
      dataset. Note that this is only done if a `train_eval_fn` argument is
      supplied.
    private_rounds_per_public_rounds: The number of rounds sampling private
    clients per 1 round of sampling public clients.
    server_state_epoch_update_fn: A function to update the `SeverState` outside
      of TFF iterative process. It is called at the beginning of each epoch
      traversing all the clients. Used to restart tree for FTRL algorithm.

  Returns:
    The final `state` of the iterative process after training.
  """
  if not isinstance(iterative_process, tff.templates.IterativeProcess):
    raise TypeError('iterative_process should be type ')
  if not callable(client_datasets_fn):
    raise TypeError('client_datasets_fn should be callable.')
  if not callable(validation_fn):
    raise TypeError('validation_fn should be callable.')
  if train_eval_fn is not None and not callable(train_eval_fn):
    raise TypeError('train_eval_fn should be callable.')
  if test_fn is not None and not callable(test_fn):
    raise TypeError('test_fn should be callable.')

  logging.info('Starting iterative_process training loop...')
  initial_state = iterative_process.initialize()

  checkpoint_mngr, metrics_mngr, tensorboard_mngr, _ = training_loop._setup_outputs(
      root_output_dir, experiment_name, hparam_dict)

  logging.info('Asking checkpoint manager to load checkpoint.')
  state, round_num = checkpoint_mngr._load_checkpoint_from_path(
      initial_state,
      warmstart_file)
  logging.info('Finished loading warmstarted checkpoint from {}'.format(warmstart_file))

  epoch = 0 if total_epochs > 0 else -1
  if state is None or total_epochs > 0:
    state = initial_state
    round_num = 0
    logging.info('Initializing experiment from scratch at round %d.', round_num)
  else:
    logging.info('Restarted from checkpoint round %d', round_num)
    round_num += 1  # Increment to avoid overwriting current checkpoint
  metrics_mngr.clear_metrics(round_num)

  loop_start_time = time.time()
  while epoch < total_epochs and round_num < total_rounds:
    data_prep_start_time = time.time()
    prev_epoch = epoch

    federated_train_data, epoch = client_datasets_fn(round_num, epoch)

    if server_state_epoch_update_fn is not None and epoch == prev_epoch + 1:
      logging.info('External server state update at epoch %d', epoch)
      state = server_state_epoch_update_fn(state)

    train_metrics = {
        'prepare_datasets_secs': time.time() - data_prep_start_time
    }
    training_start_time = time.time()

    state, _ = iterative_process.next(state, federated_train_data)
    train_metrics['training_secs'] = time.time() - training_start_time

    logging.info('Round {:2d}, {:.2f}s per round in average.'.format(
        round_num, (time.time() - loop_start_time) / (round_num + 1)))

    if (round_num % rounds_per_checkpoint == 0 or
        round_num == total_rounds - 1):
      save_checkpoint_start_time = time.time()
      try:
        checkpoint_mngr.save_checkpoint(state, round_num)
      except Exception:  # pylint: disable=broad-except
        logging.info('Checkpoint saving exception: %s', Exception)
      train_metrics['save_checkpoint_secs'] = (
          time.time() - save_checkpoint_start_time)

    metrics = {'train': train_metrics}

    if train_eval_fn and round_num % rounds_per_train_eval == 0:
      # Compute metrics over the entire training dataset
      train_eval_start = time.time()
      train_eval_metrics = train_eval_fn(state.model)
      train_eval_metrics['evaluate_secs'] = time.time() - train_eval_start
      metrics['train_eval'] = train_eval_metrics

    if round_num % rounds_per_eval == 0:
      # Compute validation metrics
      evaluate_start_time = time.time()
      validation_metrics = validation_fn(state.model)
      validation_metrics['evaluate_secs'] = time.time() - evaluate_start_time
      metrics['eval'] = validation_metrics
      training_loop._write_metrics(metrics_mngr, tensorboard_mngr, metrics,
                                   round_num)

    round_num += 1

  # Final metrics evaluation once the training has completed
  metrics = {}

  # Validation metrics
  evaluate_start_time = time.time()
  validation_metrics = validation_fn(state.model)
  validation_metrics['evaluate_secs'] = time.time() - evaluate_start_time
  metrics['eval'] = validation_metrics

  # Training set metrics
  if train_eval_fn:
    train_eval_start = time.time()
    train_eval_metrics = train_eval_fn(state.model)
    train_eval_metrics['evaluate_secs'] = time.time() - train_eval_start
    metrics['train_eval'] = train_eval_metrics

  # Test set metrics
  if test_fn:
    test_start_time = time.time()
    test_metrics = test_fn(state.model)
    test_metrics['evaluate_secs'] = time.time() - test_start_time
    metrics['test'] = test_metrics
  training_loop._write_metrics(metrics_mngr, tensorboard_mngr, metrics,
                               round_num)

  return state
