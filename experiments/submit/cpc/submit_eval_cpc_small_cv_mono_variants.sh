#!/bin/bash
#SBATCH --account=ank@gpu
#SBATCH --output=logs/eval_cpc_small_cv-mono_%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --array=0-260
#SBATCH --hint=nomultithread          # hyperthreading desactive


echo SBATCH --array=0-260%60



# This script submits CPC small training experiments.
# It will submit 1 job per line of experiments_txt/cpc_small_experiments.txt
# Arguments of trainers/train_cpc_small.sh must be lines of experiments_txt/cpc_small_experiments.txt

source activate inftrain
module load sox

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /gpfsdswork/projects/rech/ank/ucv88ce/projects/MultilingualCPC/experiments/experiments_txt_mono/cpc_small_experiments.txt)


NCLUSTERS=50
CONTRAST=fr_ii-uu
LANG=fr

cd ..
./evaluators/evaluate_cpc_cv_mono_variants.sh ${ARGS} ${NCLUSTERS} ${CONTRAST} ${LANG}



