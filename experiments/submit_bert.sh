#!/bin/bash
#SBATCH --output=logs/lstm_%A_%a.out
#SBATCH --account=cfs@gpu
#SBATCH --mem=128G
#SBATCH -C v100-32g
#SBATCH --array=1-1%254

# This script submits LSTM training experiments.
# It will submit 1 job per line of experiments_txt/lstm_experiments.txt
# Arguments of trainers/train_lstm.sh must be lines of experiments_txt/lstm_experiments.txt

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /gpfsscratch/rech/cfs/uow84uh/InfTrain/experiments/experiments_txt/bert_experiments.txt)
cd ..
./trainers/train_bert_big.sh ${ARGS}
