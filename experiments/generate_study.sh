#!/bin/bash

#if [ $IDRPROJ != cfs ]; then
#  echo 'You should run "eval $(idrenv -d cfs)" before running this script.'
#  exit
#fi;

PARAMS_FILE=infants_training.txt
DATA_PATH=blabla
LANGUAGES=(English French)
SIZES=(50 100 200 400 800 1600 3200)

rm -f $PARAMS_FILE

for (( POWER=6 ; POWER >= 0 ; POWER-- )); do                                # loop through size
  SIZE=$((50*2**$((6-POWER))))
  for (( SHARE=0 ; SHARE < $((2**POWER)) ; SHARE++ )); do                   # loop through number of shares
    for LANGUAGE in ${LANGUAGES[*]}; do
      MODEL_PATH=${ALL_CCFRSCRATCH}/experiments/${LANGUAGE}/${SIZE}h/share${SHARE}
    done;
  done;
done;
