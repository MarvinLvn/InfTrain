#!/usr/bin/env bash
#SBATCH --account=cfs@gpu
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1                # nombre de GPU a reserver
#SBATCH --cpus-per-task=10          # nombre de cpus par GPU : rule of thumb n_cpu = 10*n_gpus
#SBATCH --hint=nomultithread

PATH_DB=$1
PATH_CPC=$2
PATH_CPT=deduce from PATH_CPC
LEVEL_GRU=deduce from PATH_CPC
NB_EPOCHS=deduce from path db

if [ -f ${PATH_CPT}/running.state ]; then
  echo "${PATH_CPT}/running.state found. Not running anything."
  exit
fi;

touch ${PATH_CPT}/running.state
python CPC_audio/cpc/criterion/clustering/clustering_script.py --recursionLevel 2 --extension wav \
        --nClusters 50 --MAX_ITER 150 --save --batchSizeGPU 200 --level_gru ${LEVEL_GRU} --save --load \
        ${PATH_CPC} ${PATH_OUTPUT_CHECKPOINT}

rm ${PATH_CPT}/running.state
if [ -f ${PATH_CPT}/checkpoint${NB_EPOCHS}.pt ]; then
  touch ${PATH_CPT}/done.state
fi;