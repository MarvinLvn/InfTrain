#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --output=../logs/eval_cpc_small_gpu_%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --array=0-120%60
#SBATCH --hint=nomultithread          # hyperthreading desactive




# This script submits CPC small training experiments.
# It will submit 1 job per line of experiments_txt/cpc_small_experiments.txt
# Arguments of trainers/train_cpc_small.sh must be lines of experiments_txt/cpc_small_experiments.txt

source activate inftrain
module load sox

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /gpfsdswork/projects/rech/ank/ucv88ce/projects/MultilingualCPC/experiments/experiments_txt/cpc_small_experiments.txt)
cd ../..
./evaluators/evaluate_cpc_gpu.sh ${ARGS}


