#!/bin/bash
source ~/.bashrc
source activate public-data-in-dpfl

DATE=`date +%Y-%m-%d-%T`
echo $DATE

# blaze run experimental/users/vinithms/public_dpfl:run_dpfl_federated_research \
# --server_optimizer=dpsgdm \
# --experiment_type=private \
# --dataset=stackoverflow \
# --lstm_cell=LSTM \
# --total_epochs=1 \
# --rounds_per_eval=1 \
# --clients_per_round=3 \
# --client_batch_size=4 \
# --total_rounds=10 \
# --max_elements_per_user=16 \
# --rounds_per_eval=2 \


# python -u run_dpfl.py \
# --server_optimizer=dpsgdm \
# --experiment_type=public_SO \
# --dataset=stackoverflow \
# --noise_multiplier=0.0 \
# --lstm_cell=LSTM \
# --server_lr=3.0 \
# --client_lr=0.5 \
# --clip_norm=1.0 \
# --rounds_per_eval=1 \
# --clients_per_round=100 \
# --private_round_size=100 \
# --public_round_size=100 \
# --client_batch_size=4 \
# --total_rounds=30 \
# --max_elements_per_user=16 \
# --rounds_per_eval=5 \
# --root_output_dir=/scratch/gobi2/vinithms/public-data-in-dpfl/public_SO_v4/ \


# python -u ./run_dpfl.py \
# --server_optimizer=dpsgdm \
# --experiment_type=warmstart \
# --dataset=stackoverflow \
# --noise_multiplier=0.4 \
# --server_lr=3.0 \
# --client_lr=0.5 \
# --clip_norm=1.0 \
# --clients_per_round=100 \
# --private_round_size=100 \
# --public_round_size=100 \
# --client_batch_size=16 \
# --total_rounds=10 \
# --max_elements_per_user=16 \
# --rounds_per_eval=5 \
# --warmstart_file=/scratch/gobi2/vinithms/public-data-in-dpfl/public_SO_v2/checkpoints/stackoverflow/ckpt_29 \
# --root_output_dir=/scratch/gobi2/vinithms/public-data-in-dpfl/dp_fedavg_from_public_SO_v22/ \


python -u ./run_dpfl.py \
--server_optimizer=dpsgdm \
--experiment_type=mirror_descent_convex_warmstart_SO \
--server_optimizer=dpsgdm \
--noise_multiplier=0.4 \
--server_lr=3.0 \
--client_lr=0.5 \
--clip_norm=1.0 \
--clients_per_round=100 \
--private_round_size=100 \
--public_round_size=100 \
--client_batch_size=16 \
--total_rounds=10 \
--max_elements_per_user=16 \
--rounds_per_eval=5 \
--warmstart_file=/scratch/gobi2/vinithms/public-data-in-dpfl/public_SO_v2/checkpoints/stackoverflow/ckpt_29 \
--root_output_dir=/scratch/gobi2/vinithms/public-data-in-dpfl/dp_md_real_16_v16_from_public_SO/ \


# python -u ./run_dpfl_md.py \
# --server_optimizer=dpsgdm \
# --experiment_type=mirror_descent_convex_SO \
# --server_optimizer=dpsgdm \
# --noise_multiplier=0.0 \
# --server_lr=3 \
# --client_lr=0.5 \
# --clip_norm=1.0 \
# --rounds_per_eval=1 \
# --public_round_size=3 \
# --private_round_size=3 \
# --total_rounds=25 \
# --max_elements_per_user=16 \
# --rounds_per_eval=5 \