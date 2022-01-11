#!/bin/bash
#SBATCH --output=logs/eval_cpc_big_%A_%a.out
#SBATCH --partition=devlab
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --array=253-254%64

# This script submits CPC big evaluation experiments.
# It will submit 1 job per line of experiments_txt/cpc_big_experiments.txt
# Arguments of evaluators/evaluate_cpc.sh must be lines of experiments_txt/cpc_big_experiments.txt

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /private/home/marvinlvn/InfTrain/fair_submit/experiments/experiments_txt/cpc_big_experiments.txt)
cd ..
./evaluators/evaluate_cpc_ls.sh ${ARGS}