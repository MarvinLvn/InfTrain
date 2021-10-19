#!/bin/bash
#SBATCH --account=ank@gpu
#SBATCH --output=../logs/cpc_small_%A_%a.out
#SBATCH --nodes=2                     # nombre de noeud
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4                  # nombre de GPUs par nœud
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --array=1-6%60

# This script submits CPC small training experiments.
# It will submit 1 job per line of experiments_txt/cpc_small_experiments.txt
# Arguments of trainers/train_cpc_small.sh must be lines of experiments_txt/cpc_small_experiments.txt

echo "TASK ID : $SLURM_ARRAY_TASK_ID"
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /gpfsdswork/projects/rech/ank/ucv88ce/projects/MultilingualCPC/experiments/experiments_txt/cpc_small_experiments.txt)

cd ../..
./trainers/train_cpc_small.sh ${ARGS}
