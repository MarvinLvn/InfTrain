#!/bin/bash
#SBATCH --partition=learnfair
#SBATCH --output=logs/cpc_small_%A_%a.out
#SBATCH --nodes=2                     # nombre de noeud
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4                  # nombre de GPUs par nœud
#SBATCH --time=72:00:00
#SBATCH --array=1-1%64

# This script submits CPC small training experiments.
# It will submit 1 job per line of experiments_txt/cpc_small_experiments.txt
# Arguments of trainers/train_cpc_small.sh must be lines of experiments_txt/cpc_small_experiments.txt

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /private/home/marvinlvn/InfTrain/fair_submit/experiments/experiments_txt/cpc_small_experiments.txt)
cd ..
./trainers/train_cpc_small.sh ${ARGS}