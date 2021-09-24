#!/bin/bash

#if [ $IDRPROJ != cfs ]; then
#  echo 'You should run "eval $(idrenv -d cfs)" before running this script.'
#  exit
#fi;

DATA_PATH=/gpfsscratch/rech/cfs/commun/families
LANGUAGES=(EN FR)

rm -f $PARAMS_FILE
mkdir -p last_experiments_txt
mkdir -p experiments_txt
mv -f experiments_txt/{cpc_small,cpc_big,kmeans,lstm,bert_large}_experiments.txt last_experiments_txt
touch experiments_txt/{cpc_small,cpc_big,kmeans,lstm,bert_large}_experiments.txt

for (( POWER=6 ; POWER >= 0 ; POWER-- )); do                                # loop through size
  SIZE=$((50*2**$((6-POWER))))
  for (( SHARE=0 ; SHARE < $((2**POWER)) ; SHARE++ )); do                   # loop through number of shares
    printf -v SHARE_NB "%02d" $SHARE
    for LANGUAGE in ${LANGUAGES[*]}; do
      MODEL_PATH=${ALL_CCFRSCRATCH}/InfTrain_models/${LANGUAGE}/${SIZE}h/${SHARE_NB}
      PATH_DB=${ALL_CCFRSCRATCH}/families/${LANGUAGE}/${SIZE}h/${SHARE_NB}
      ./which_model.sh $MODEL_PATH $PATH_DB
    done;
  done;
done;

echo "Done writing experiment files in experiments_txt/"

