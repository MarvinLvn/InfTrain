#!/bin/bash

#if [ $IDRPROJ != cfs ]; then
#  echo 'You should run "eval $(idrenv -d cfs)" before running this script.'
#  exit
#fi;

DATA_PATH=blabla
LANGUAGES=(English French)

rm -f $PARAMS_FILE
mkdir -p last_experiments_txt
mkdir -p experiments_txt
mv -f experiments_txt/{cpc_small,cpc_big,kmeans,lstm,bert_large}_experiments.txt last_experiments_txt
touch experiments_txt/{cpc_small,cpc_big,kmeans,lstm,bert_large}_experiments.txt

for (( POWER=6 ; POWER >= 0 ; POWER-- )); do                                # loop through size
  SIZE=$((50*2**$((6-POWER))))
  for (( SHARE=0 ; SHARE < $((2**POWER)) ; SHARE++ )); do                   # loop through number of shares
    for LANGUAGE in ${LANGUAGES[*]}; do
      MODEL_PATH=${ALL_CCFRSCRATCH}/experiments/${LANGUAGE}/${SIZE}h/share${SHARE}
      ./which_model.sh $MODEL_PATH
    done;
  done;
done;
