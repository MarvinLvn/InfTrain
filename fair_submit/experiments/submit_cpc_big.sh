#!/bin/bash
#SBATCH --partition=learnlab
#SBATCH --output=logs/cpc_big_%A_%a.out
#SBATCH --nodes=4
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # nombre de GPUs par nœud
#SBATCH --time=72:00:00
#SBATCH --constraint volta32gb
#SBATCH --array=253-254%64

# This script submits CPC big training experiments.
# It will submit 1 job per line of experiments_txt/cpc_big_experiments.txt
# Arguments of trainers/train_cpc_big.sh must be lines of experiments_txt/cpc_big_experiments.txt

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /private/home/marvinlvn/InfTrain/fair_submit/experiments/experiments_txt/cpc_big_experiments.txt)
cd ..
./trainers/train_cpc_big.sh ${ARGS}
