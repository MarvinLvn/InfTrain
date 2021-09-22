#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "./which_model.sh /model/path /database/path"
    exit
fi

# We're gonna need MODEL_PATH at some point.
# We need to talk about which dataset will be the training set of kmeans model/lm.
MODEL_PATH=$1
PATH_DB=$2
SHARE=$(basename $MODEL_PATH)
SHARE=${SHARE/share/}
SIZE=$(basename $(dirname $MODEL_PATH))
SIZE=${SIZE/h/}

CPC="cpc_small"
LM="lstm"
KMEANS="kmeans_50"

# We changed our plan : we only want to train the small arch. for now
#if [ $SIZE -ge 800 ]; then
#  CPC="cpc_big"
#  LM="bert_large"
#fi;

# Train CPC
if [ ! -f ${MODEL_PATH}/cpc_small/done.state ]; then
  # Redirect to cpc_small_experiments.txt and cpc_big_experiments.txt
  echo ${PATH_DB} >> experiments_txt/cpc_small_experiments.txt
fi;
if [ ! -f ${MODEL_PATH}/cpc_big/done.state ]; then
  echo ${PATH_DB} >> experiments_txt/cpc_big_experiments.txt
fi;
if [ ! -f ${MODEL_PATH}/${KMEANS}/done.state ]; then
  # Redirect to kmeans_experiments.txt
  echo ${PATH_DB} >> experiments_txt/kmeans_experiments.txt
fi;
if [ ! -f ${MODEL_PATH}/${LM}/done.state ]; then
  # Redirect to lstm_experiments.txt and bert_large_experiments.txt
  echo ${PATH_DB} >> experiments_txt/${LM}_experiments.txt
fi;
