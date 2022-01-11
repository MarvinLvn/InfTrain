#!/bin/bash
#SBATCH --partition=learnlab
#SBATCH --output=logs/cpc_big_kmeans_%A_%a.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1                  # nombre de GPUs par nœud
#SBATCH --time=72:00:00

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

CPC_NAME=cpc_big
NB_EPOCHS=300

PATH_CPT=/checkpoint/marvinlvn/InfTrain/InfTrain_models/${LANGUAGE}/${SIZE}/${SHARE}/kmeans50
PATH_CPC=/checkpoint/marvinlvn/InfTrain/InfTrain_models/${LANGUAGE}/${SIZE}/${SHARE}/${CPC_NAME}
BEST_EPOCH=$(python $HOME/CPC_audio_jzay/utils/best_val_epoch.py --model_path ${PATH_CPC} | grep -oP "(?<=is : )([0-9]+)")
PATH_CPC=${PATH_CPC}/checkpoint_${BEST_EPOCH}.pt

LEVEL_GRU=2

#mkdir -p ${PATH_CPT}
#touch ${PATH_CPT}/running.state
echo "Start training $PATH_CPT"

python /private/home/marvinlvn/InfTrain/CPC_torch/cpc/clustering/clustering_script.py ${PATH_CPC} ${PATH_CPT} ${PATH_DB} --recursionLevel 2 --extension wav \
        --nClusters 50 --MAX_ITER $NB_EPOCHS --save --batchSizeGPU 200 --level_gru ${LEVEL_GRU} --perIterSize 1406 --save-last 5