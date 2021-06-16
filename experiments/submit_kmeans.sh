#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1                # nombre de GPU a reserver
#SBATCH --cpus-per-task=10          # nombre de cpus par GPU : rule of thumb n_cpu = 10*n_gpus
#SBATCH --output=logs/kmeans/%A_%a.out
#SBATCH --hint=nomultithread
#SBATCH --array=1-300%150


ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p noise_study_parameters.txt)
echo $ARGS

./train_CPC.sh ${ARGS}
