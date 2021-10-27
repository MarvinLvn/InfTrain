#!/bin/bash
#SBATCH --account=ank@gpu
#SBATCH --output=logs/eval_kmeans_cv_%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --array=0-130%60
#SBATCH --hint=nomultithread          # hyperthreading desactive

# This script submits CPC small training experiments.
# It will submit 1 job per line of experiments_txt/cpc_small_experiments.txt
# Arguments of trainers/train_cpc_small.sh must be lines of experiments_txt/cpc_small_experiments.txt



ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /gpfsdswork/projects/rech/ank/ucv88ce/projects/MultilingualCPC/experiments/experiments_txt/kmeans_experiments.txt)
cd ..

NCLUSTERS=50
CONTRAST=fr_ii-uu
LANG=fr

./evaluators/evaluate_kmeans_cv_variants.sh ${ARGS} ${NCLUSTERS} ${CONTRAST} ${LANG}
