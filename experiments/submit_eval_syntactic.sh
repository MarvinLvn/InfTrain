#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --output=logs/eval_syntactic_%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --array=1-254%254
#SBATCH --hint=nomultithread          # hyperthreading desactive

source activate inftrain
module load sox

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /gpfsscratch/rech/cfs/uow84uh/InfTrain/experiments/experiments_txt/lstm_experiments.txt)
cd ..
./evaluators/evaluate_lm_syntactic.sh ${ARGS}

