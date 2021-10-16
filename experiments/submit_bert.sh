#!/bin/bash
#SBATCH --output=logs/bert_%A_%a.out
#SBATCH --partition=gpu_p2
#SBATCH --account=cfs@gpu
#SBATCH --nodes=4                     # nombre de noeud
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # nombre de GPUs par nœud
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --array=253-254%254
#SBATCH --time=20:00:00

# This script submits LSTM training experiments.
# It will submit 1 job per line of experiments_txt/lstm_experiments.txt
# Arguments of trainers/train_lstm.sh must be lines of experiments_txt/lstm_experiments.txt

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /gpfsscratch/rech/cfs/uow84uh/InfTrain/experiments/experiments_txt/bert_experiments.txt)
cd ..
./trainers/train_bert_big.sh ${ARGS}
