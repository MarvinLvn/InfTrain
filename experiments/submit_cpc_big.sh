#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --output=logs/cpc_big_%A_%a.out
#SBATCH --partition=gpu_p2            # access to octo-gpus machines
#SBATCH --nodes=4                     # nombre de noeud
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8                  # nombre de GPUs par nœud
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --exclusive
#SBATCH --array=0-14%14

# This script submits CPC big training experiments.
# It will submit 1 job per line of experiments_txt/cpc_big_experiments.txt
# Arguments of trainers/train_cpc_big.sh must be lines of experiments_txt/cpc_big_experiments.txt

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /gpfsscratch/rech/cfs/uow84uh/InfTrain/experiments/experiments_txt/cpc_big_experiments.txt)
cd ..
./trainers/train_cpc_big.sh ${ARGS}