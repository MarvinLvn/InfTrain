#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --output=../logs/lstm_eval_prosody1_%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --array=253-254%254
#SBATCH --hint=nomultithread          # hyperthreading desactive

source activate inftrain
module load sox

if [[ $# -ne 1 ]]; then
  echo "sbatch submit_eval_prosody1.sh bert_sbm_none"
fi;

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /gpfsscratch/rech/cfs/uow84uh/InfTrain/experiments/experiments_txt/bert_experiments.txt)
cd ../..
./evaluators/evaluate_lm_prosody.sh ${ARGS} $1
