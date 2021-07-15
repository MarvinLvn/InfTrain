#!/usr/bin/env bash
#SBATCH --account=cfs@gpu
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1                # nombre de GPU a reserver
#SBATCH --cpus-per-task=10          # nombre de cpus par GPU : rule of thumb n_cpu = 10*n_gpus
#SBATCH --hint=nomultithread

PATH_DB=$1

if [ "$#" -ne 2 ]; then
  echo "Illegal number of parameters. Please respect :"
  echo "./train_cpc_kmeans.sh /path/to/database/containing/wav/files"
  echo "Example :"
  echo "./train_cpc_kmeans.sh /gpfsscratch/rech/cfs/commun/families/EN/50h/00"
  exit
fi

SHARE=$(basename $PATH_DB)
SIZE=$(basename $(dirname $PATH_DB))
LANGUAGE=$(basename $(dirname $(dirname $PATH_DB)))
PATH_CPT=${ALL_CCFRSCRATCH}/InfTrain_models/${LANGUAGE}/${SIZE}/${SHARE}/kmeans
PATH_CPC=deduce from PATH_DB
LEVEL_GRU=deduce from PATH_CPC
NB_EPOCHS=deduce from path db

if [ -f ${PATH_CPT}/running.state ]; then
  echo "${PATH_CPT}/running.state found. Not running anything."
  exit
fi;

mkdir -p $PATH_CPT
touch ${PATH_CPT}/running.state
python CPC_audio/cpc/criterion/clustering/clustering_script.py --recursionLevel 2 --extension wav \
        --nClusters 50 --MAX_ITER 150 --save --batchSizeGPU 200 --level_gru ${LEVEL_GRU} --save --load \
        ${PATH_CPC} ${PATH_OUTPUT_CHECKPOINT}

rm ${PATH_CPT}/running.state
if [ -f ${PATH_CPT}/checkpoint${NB_EPOCHS}.pt ]; then
  touch ${PATH_CPT}/done.state
fi;