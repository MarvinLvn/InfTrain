#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --output=../logs/eval_cpc_small_%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --array=1-1%100

# This script submits CPC small training experiments.
# It will submit 1 job per line of experiments_txt/cpc_small_experiments.txt
# Arguments of trainers/train_cpc_small.sh must be lines of experiments_txt/cpc_small_experiments.txt

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /gpfsscratch/rech/cfs/uow84uh/InfTrain/experiments/experiments_txt/cpc_eval_experiments.txt)
cd ../..
./evaluators/evaluate_cpc_ABX_cv.sh ${ARGS}
