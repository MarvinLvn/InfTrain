#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --output=logs/eval_cpc_big_%A_%a.out
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --array=0-14%14

# This script submits CPC big evaluation experiments.
# It will submit 1 job per line of experiments_txt/cpc_big_experiments.txt
# Arguments of evaluators/evaluate_cpc.sh must be lines of experiments_txt/cpc_big_experiments.txt

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /gpfsscratch/rech/cfs/uow84uh/InfTrain/experiments/experiments_txt/cpc_big_experiments.txt)
cd ..
./trainers/evaluate_cpc.sh ${ARGS}