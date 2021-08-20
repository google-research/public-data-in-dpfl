# Public Data in Differentially Private Federated Learning

This repository contains code of different methods of incorporating labelled
public data during differentially private federated learning.

## Datasets

We separate the StackOverflow data into private and public datasets. This
focuses on the setting where in-distribution public data is available.

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

Not an official Google product.
