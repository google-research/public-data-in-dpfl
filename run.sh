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

#!/bin/bash
set -x

DATE=`date +%Y-%m-%d-%T`
echo $DATE

blaze run experimental/users/vinithms/public_dpfl:run_dpfl_federated_research --
--server_optimizer=dpsgdm \
--experiment_type=private \
--dataset=stackoverflow \
--lstm_cell=LSTM \
--total_epochs=1 \
--rounds_per_eval=1 \
--clients_per_round=3 \
--client_batch_size=4 \
--total_rounds=100 \
--max_elements_per_user=16 \
--rounds_per_eval=5
