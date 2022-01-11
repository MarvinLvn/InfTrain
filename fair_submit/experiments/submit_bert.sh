#!/bin/bash
#SBATCH --output=logs/bert_cpc_big_kmeans_%A_%a.out
#SBATCH --partition=learnlab
#SBATCH --nodes=4                     # nombre de noeud
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8                  # nombre de GPUs par nœud
#SBATCH --array=253-254%64
#SBATCH --time=72:00:00

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /private/home/marvinlvn/InfTrain/fair_submit/experiments/experiments_txt/bert_experiments.txt)
cd ..
./trainers/train_bert_big.sh ${ARGS}