# Public Data in Differentially Private Federated Learning

This repository contains code of different methods of incorporating labelled
public data during differentially private federated learning.

## Datasets

We separate the StackOverflow data into private and public datasets. This
focuses on the setting where matched public data is available.

## Methods

The different methods we explored include: 1. Warmstarting a model on public
data and finetuning on private data, 2. Alternating rounds that use private and
public clients separately, 3.[SCAFFOLD](https://arxiv.org/pdf/1910.06378.pdf)
using private data for weights and public data for control variates 4.
[MIME](https://arxiv.org/pdf/2008.03606.pdf) using private data for weights and
public data for average gradients. 5. A federated variant of mirror descent
which uses private and public data.

## Structure of Repository

The `run_dpfl.py` is used to configure each experiment and run the corresponding
method based on the provided hyperparameters. This file can be run from
`run.sh`.

### Data Preprocessing

For processing the sentences into their tokenized IDs this is done in the
`data_word_prediction.py` file for both StackOverflow.

### Non-Private and DP-FedAvg Baselines

For running and making changes to the non-private and DP-FedAvg baselines this
will need to be done in the `train_and_eval` function in `run_dpfl.py`. For
`non-private` baseline this will correspond to setting the experiment type to
`private` (corresponding to the private dataset) and setting the noise
multiplier to 0.0. For the `DP-FedAvg` baseline this will correspond to setting
the experiment type to `private` and setting the noise multiplier to greater
than 0.0.

### Training Only on Public Data

For running and making changes to the training on public data baselines this
will need to be done in the `train_and_eval` function in `run_dpfl.py`. For the
`100 Public SO` baseline this will correspond to setting the experiment type to
`public_SO` and setting the noise multiplier to 0.0. For `100 Public SO` if you
would like to change the number of clients that are used you can do this by
setting `public_client_size` to whatever you would like. The default is 100.

For the Mime / MimeLite experiments because they use Keras optimizers there is
an incomptability between the original checkpoints and the implementation. Thus,
there are separate functions for creating checkpoints which are compatible with
Mime / MimeLite. These can be run by setting the experiment type to
`stackoverflow_SGD`. The code for the server and client updates can be found in
`public.py` and the training loop is `public_loop.py`.

### Alternating between Private and Public Updates

For running the alternating (mismatched and matched) experiments this is done in
the `train_and_eval_alternating` function in `run_dpfl.py`. The training loop
code for alternating experiments is in `alternating_loop.py`. The code works by
creating two separate `tff.simulation.IterativeProcess` objects -- one for the
private clients and one for the public clients. The experiments can be run by
setting the experiment type to `alternating` and `alternating_SO` for mismatched
and matched public data respectively. For alternating + warmstarting the
experiment type must be set to `alternating_warmstart` and
`alternating_warmstart_SO`.

### SCAFFOLD with Private and Public Data

For running the SCAFFOLD (mismatched and matched) experiments this is done in
the `train_and_eval_scaffold` function in `run_dpfl.py`. The training loop code
for alternating experiments is in `scaffold_loop.py`. The code works by creating
two separate `tff.simulation.IterativeProcess` objects -- one for the private
clients and one for the public clients. The actual training algorithm is
implemented in `scaffold_v2.py`. In this code there are separate client and
server update functions for the private and public clients. The experiments can
be run by setting the experiment type to `scaffold` and `scaffold_SO` for
mismatched and matched public data respectively. For scaffold + warmstarting the
experiment type must be set to `scaffold_warmstart` and `scaffold_warmstart_SO`.

### Mime / MimeLite with Private and Public Data

For running the MimeLite (mismatched and matched) experiments this is done in
the `train_and_eval_mimelite` function in `run_dpfl.py`. The training loop code
for alternating experiments is in `mimelite_loop.py`. The code works by creating
two separate `tff.simulation.IterativeProcess` objects -- one for the private
clients and one for the public clients. The actual training algorithm is
implemented in `mimelite.py`. In this code there are separate client and server
update functions for the private and public clients. The experiments can be run
by setting the experiment type to `mimelite` and `mimelite_SO` for mismatched
and matched public data respectively. For mimelite + warmstarting the experiment
type must be set to `mimelite_warmstart` and `mimelite_warmstart_SO`.

For running the Mime (mismatched and matched) experiments this is done in the
`train_and_eval_mime'` function in `run_dpfl.py`. The training loop code for
alternating experiments is in `mime_loop.py`. The code works by creating two
separate `tff.simulation.IterativeProcess` objects -- one for the private
clients and one for the public clients. The actual training algorithm is
implemented in `mime.py`. In this code there are separate client and server
update functions for the private and public clients. The experiments can be run
by setting the experiment type to `mime` and `mime_SO` for mismatched and
matched public data respectively. For mimelite + warmstarting the experiment
type must be set to `mime_warmstart` and `mime_warmstart_SO`.

### Federated Mirror Descent with Public Data

For running the federated mirror descent (mismatched and matched) experiments
this is done in the `train_and_eval_mirror_descent'` function in `run_dpfl.py`.
The training loop code for alternating experiments is in
`mirror_descent_loop.py`. The code works by creating two separate
`tff.simulation.IterativeProcess` objects -- one for the private clients and one
for the public clients. The actual training algorithm is implemented in
`mirror_descent.py`. In this code there are separate client and server update
functions for the private and public clients. The experiments can be run by
setting the experiment type to `mirror_descent` and `mirror_descent_SO` for
mismatched and matched public data respectively. For mimelite + warmstarting the
experiment type must be set to `mirror_descent_warmstart` and
`mirror_descent_warmstart_SO`.

## Adding New Methods

For adding new algorithms this is the process I would follow: 1. Define
experiment type name and add this to options in FLAGS list 2. Add an if
statement to main in run_dpfl.py to account for new method. 3. Create another
train_and_eval_<method_name> function to manage all the logic. 4. Create a data
processing function similar to '_preprocess_data' 5. Create an iterative process
+ model builder function similar to _build_custom_model_and_process 6. Create a
<method_name>_loop.py file to manage the training loop 7. Create a
<method_name>.py file to manage the algorithm logic

Not an official Google product.
