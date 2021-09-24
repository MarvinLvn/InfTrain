#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --output=logs/kmeans_%A_%a.out
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1                # nombre de GPU a reserver
#SBATCH --cpus-per-task=10          # nombre de cpus par GPU : rule of thumb n_cpu = 10*n_gpus
#SBATCH --hint=nomultithread
#SBATCH --array=1-254%60

# This script submits K-means training experiments.
# It will submit 1 job per line of experiments_txt/kmeans_experiments.txt
# Arguments of trainers/train_cpc_small.sh must be lines of experiments_txt/cpc_small_experiments.txt

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /gpfsscratch/rech/cfs/uow84uh/InfTrain/experiments/experiments_txt/kmeans_experiments.txt)
cd ..
./trainers/train_kmeans.sh ${ARGS}
