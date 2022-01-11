# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################
# You will find here an example of how to run the distributed mode
# on the FAIR cluster. This is extremly useful for big datasets
#####################################################################

import submitit
import os
from pathlib import Path

#####################################################################

SLURM_LOGS_DIR = "/private/home/marvinlvn/InfTrain/fair_submit/experiments/logs"
CHECKPOINT_DIR = "/checkpoint/marvinlvn/InfTrain/InfTrain_models/EN/50h/05/cpc_big"
PATH_DB = "/private/home/marvinlvn/DATA/CPC_data/train/InfTrain/EN/50h/05"

#####################################################################

os.makedirs(SLURM_LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

args = ['--hiddenEncoder', '512',
        '--hiddenGar', '512',
        '--dropout',
        '--nEpoch', '10',
        '--nLevelsGRU', '4',
        '--schedulerRamp', '10',
        '--pathDB', str(PATH_DB),
        '--pathCheckpoint', str(CHECKPOINT_DIR),
        '--rnnMode', "transformer",
        '--samplingType', "samespeaker",
        '--save_step', '1',
        '--distributed',
        '--nGPU', '1',
        '--batchSizeGPU', '16',
        '--master_port', '18363',
        '--file_extension', ".wav",
        '--restart',]

# submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder=str(SLURM_LOGS_DIR))
executor.update_parameters(timeout_min=60 * 24 * 3, mem_gb=128,
                           gpus_per_node=8, tasks_per_node=8, nodes=4, cpus_per_task=10,
                           slurm_partition="learnlab",
                           slurm_comment='Training of CPC big on InfTrain', name='CPC_big')

def main(args):
    import sys
    sys.path.append('../..')
    import cpc.train
    return cpc.train.main(args)

job = executor.submit(main, args)
print(f"Slurm job submitted. ID: {job.job_id}")
