#!/bin/bash
#SBATCH --partition=devlab
#SBATCH --output=logs/cpc_big_kmeans_%A_%a.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1                  # nombre de GPUs par nœud
#SBATCH --time=72:00:00
#SBATCH --array=253-254%64


ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /private/home/marvinlvn/InfTrain/fair_submit/experiments/experiments_txt/kmeans_experiments.txt)
cd ..
./trainers/train_kmeans.sh ${ARGS}
