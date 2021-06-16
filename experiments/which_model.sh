#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit
fi

MODEL_PATH=$1
SHARE=$(basename $MODEL_PATH)
SHARE=${SHARE/share/}
SIZE=$(basename $(dirname $MODEL_PATH))
SIZE=${SIZE/h/}

CPC="cpc_small"
LM="lstm"
KMEANS="kmeans_50"
if [ $SIZE -ge 800 ]; then
  CPC="cpc_big"
  LM="bert_large"
fi;

EXPERIMENT="cpc_small "

# Train CPC
if [ ! -f ${MODEL_PATH}/{CPC}/done.state ]; then
  # Redirect to cpc_small_experiments.txt and cpc_big_experiments.txt
  echo ${MODEL_PATH}/${CPC} >> experiments_txt/${CPC}_experiments.txt
  exit
fi;

# Train KMEANS
if [ ! -f ${MODEL_PATH}/{KMEANS}/done.state ]; then
  # Redirect to kmeans_experiments.txt
  echo ${MODEL_PATH}/${KMEANS} >> experiments_txt/kmeans_experiments.txt
  exit
fi;

# Train LM
if [ ! -f ${MODEL_PATH}/{LM}/done.state ]; then
  # Redirect to lstm_experiments.txt and bert_large_experiments.txt
  echo ${MODEL_PATH}/${LM} >> experiments_txt/${LM}_experiments.txt
  exit
fi;
