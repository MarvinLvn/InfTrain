#!/usr/bin/env bash
#SBATCH --account=cfs@gpu
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1                # nombre de GPU a reserver
#SBATCH --cpus-per-task=10          # nombre de cpus par GPU : rule of thumb n_cpu = 10*n_gpus
#SBATCH --hint=nomultithread

# Need to decide if we train on a subset : talk with Emmanuel
echo "This script hasn't been tested. Needs to be finished and thoroughly checked"
exit

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


if [ "$SIZE" == "50h" ]; then
  NB_EPOCHS=200
  CPC_NAME=cpc_small
elif [ "$SIZE" == "100h" ]; then
  # 100 epochs for a 100h training set seems fine to me
  NB_EPOCHS=100
  CPC_NAME=cpc_small
elif [ "$SIZE" == "200h" ]; then
  NB_EPOCHS=80
  CPC_NAME=cpc_small
elif [ "$SIZE" == "400h" ]; then
  NB_EPOCHS=50;
  CPC_NAME=cpc_small
elif [ "$SIZE" == "800h" ]; then
  NB_EPOCHS=100
  CPC_NAME=cpc_big
elif [ "$SIZE" == "1600h" ]; then
  NB_EPOCHS=80
  CPC_NAME=cpc_big
elif [ "$SIZE" == "3200h" ]; then
  # CPC big has been trained on 32 GPUS for 30 epochs
  NB_EPOCHS=50;
  CPC_NAME=cpc_big
else
  echo "Not possible to deduce the number of epochs from the size of the training set."
  echo "You should check that you haven't called train_cpc_big.sh with a training set whose size is lower or equal than 200h"
  exit
fi;
else
  echo "Not possible to deduce the number of epochs from the size of the training set."
  echo "You should check that you haven't called train_cpc_small.sh with a training set whose size is greater or equal than 800h"
  exit
fi;

PATH_CPT=${ALL_CCFRSCRATCH}/InfTrain_models/${LANGUAGE}/${SIZE}/${SHARE}/kmeans
PATH_CPC=${ALL_CCFRSCRATCH}/InfTrain_models/${LANGUAGE}/${SIZE}/${SHARE}/cpc
LEVEL_GRU=2

if [ -f ${PATH_CPT}/running.state ]; then
  echo "${PATH_CPT}/running.state found. Not running anything."
  exit
fi;

mkdir -p $PATH_CPT
touch ${PATH_CPT}/running.state
echo "Start training $PATH_CPT"
module load sox
python /gpfsscratch/rech/cfs/uow84uh/InfTrain/CPC_torch/cpc/criterion/clustering/clustering_script.py --recursionLevel 2 --extension wav \
        --nClusters 50 --MAX_ITER 150 --save --batchSizeGPU 200 --level_gru ${LEVEL_GRU} --save --load \
        ${PATH_CPC} ${PATH_OUTPUT_CHECKPOINT}

rm ${PATH_CPT}/running.state
if [ -f ${PATH_CPT}/checkpoint${NB_EPOCHS}.pt ]; then
  touch ${PATH_CPT}/done.state
fi;