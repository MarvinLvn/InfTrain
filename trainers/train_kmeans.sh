#!/usr/bin/env bash
#SBATCH --account=cfs@gpu
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

PATH_DB=$1

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters. Please respect :"
  echo "./train_cpc_kmeans.sh /path/to/database/containing/wav/files"
  echo "Example :"
  echo "./train_cpc_kmeans.sh /gpfsscratch/rech/cfs/commun/families/EN/50h/00"
  exit
fi

SHARE=$(basename $PATH_DB)
SIZE=$(basename $(dirname $PATH_DB))
LANGUAGE=$(basename $(dirname $(dirname $PATH_DB)))

CPC_NAME=cpc_small
NB_EPOCHS=300

PATH_CPT=${ALL_CCFRSCRATCH}/InfTrain_models/${LANGUAGE}/${SIZE}/${SHARE}/kmeans50
PATH_CPC=${ALL_CCFRSCRATCH}/InfTrain_models/${LANGUAGE}/${SIZE}/${SHARE}/${CPC_NAME}
BEST_EPOCH=$(python /gpfsscratch/rech/cfs/uow84uh/InfTrain/utils/best_val_epoch.py --model_path ${PATH_CPC} | grep -oP "(?<=is : )([0-9]+)")
PATH_CPC=${PATH_CPC}/checkpoint_${BEST_EPOCH}.pt

LEVEL_GRU=2

#mkdir -p ${PATH_CPT}
#touch ${PATH_CPT}/running.state
echo "Start training $PATH_CPT"

python /gpfsscratch/rech/cfs/uow84uh/InfTrain/CPC_torch/cpc/clustering/clustering_script.py ${PATH_CPC} ${PATH_CPT} ${PATH_DB} --recursionLevel 2 --extension wav \
        --nClusters 50 --MAX_ITER $NB_EPOCHS --save --batchSizeGPU 200 --level_gru ${LEVEL_GRU} --perIterSize 1406 --save-last 5

