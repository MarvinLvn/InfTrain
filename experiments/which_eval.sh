#!/bin/bash

BEST_VAL_SCRIPT=/gpfsscratch/rech/cfs/uow84uh/nick_temp/InfTrain/utils/best_val_epoch.py
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "./which_eval.sh /model/path /database/path"
    exit
fi

MODEL_PATH=$1
PATH_DB=$2
SHARE=$(basename $MODEL_PATH)
SHARE=${SHARE/share/}
SIZE=$(basename $(dirname $MODEL_PATH))
SIZE=${SIZE/h/}

CPC="cpc_small"
LM="lstm"
KMEANS="kmeans_50"
#if [ $SIZE -ge 800 ]; then
#  CPC="cpc_big"
#  LM="bert_large"
#fi;

if [ -d ${MODEL_PATH}/$CPC ]; then
  BEST_EPOCH=$(python $BEST_VAL_SCRIPT --model_path ${MODEL_PATH}/${CPC} | grep -oP "(?<=is : )([0-9]+)")
  if [ ! -d ${MODEL_PATH}/${CPC}/ABX_CV/${BEST_EPOCH} ]; then
    echo ${PATH_DB} >> experiments_txt/cpc_eval_experiments.txt
  fi;
fi;

# Need to do the same for K-means, and language models
