#!/bin/bash
#SBATCH --account=cfs@v100
#SBATCH --output=../logs/eval_loss_commonvoice_kmeans_%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --array=1-1%60
#SBATCH --hint=nomultithread          # hyperthreading desactive

source activate inftrain
module load sox

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /gpfsscratch/rech/cfs/uow84uh/InfTrain/experiments/experiments_txt/kmeans_eval_experiments.txt)
cd ../..
./evaluators/compute_KMEANS_loss.sh ${ARGS}