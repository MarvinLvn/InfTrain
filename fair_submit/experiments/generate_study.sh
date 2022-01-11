#!/bin/bash



DATA_PATH=/private/home/marvinlvn/DATA/CPC_data/train/InfTrain
LANGUAGES=(EN FR)

rm -f $PARAMS_FILE
mkdir -p last_experiments_txt
mkdir -p experiments_txt
mv -f experiments_txt/{cpc_small,cpc_big,kmeans,lstm,bert_large}_experiments.txt last_experiments_txt
touch experiments_txt/{cpc_small,cpc_big,kmeans,lstm,bert_large}_experiments.txt

for (( POWER=6 ; POWER >= 0 ; POWER-- )); do                                # loop through size
  SIZE=$((50*2**$((6-POWER))))
  for (( SHARE=0 ; SHARE < $((2**POWER)) ; SHARE++ )); do                   # loop through number of shares
    for LANGUAGE in ${LANGUAGES[*]}; do
      printf -v SHARE_NB "%02d" $SHARE
      MODEL_PATH=/checkpoint/marvinlvn/InfTrain/InfTrain_models/${LANGUAGE}/${SIZE}h/${SHARE_NB}
      PATH_DB=${DATA_PATH}/${LANGUAGE}/${SIZE}h/${SHARE_NB}
      ./which_model.sh $MODEL_PATH $PATH_DB
    done;
  done;
done;

echo "Done writing experiment files in experiments_txt/"