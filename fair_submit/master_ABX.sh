#!/bin/bash

# Marvin : you should check if that script is working.
# Thanks :)
shopt -s expand_aliases
alias stool='/private/home/mriviere/FairInternal/stool/stool.py'

ZR_PATH=/private/home/marvinlvn/DATA/CPC_data/test/zerospeech2017/data/test
#TEST_LANGUAGES=(english french)
TEST_LANGUAGES=(english)
FILE_EXT=.wav

# Evalute CPC_small
for i in {04..06}; do
  for TEST_LANGUAGE in ${TEST_LANGUAGES[*]}; do
    PATH_ITEM_FILE=$ZR_PATH/${TEST_LANGUAGE}/1s/1s.item
    ZEROSPEECH_DATASET=$ZR_PATH/${TEST_LANGUAGE}/1s

    # CPC_torch same config than the one used on jean zay
    PATH_CPT=/checkpoint/marvinlvn/InfTrain/InfTrain_models/EN/50h/${i}/cpc_small
    BEST_EPOCH=$(python $HOME/CPC_audio_jzay/utils/best_val_epoch.py --model_path ${PATH_CPT} | grep -oP "(?<=is : )([0-9]+)")
    MODEL_PATH=${PATH_CPT}/checkpoint_${BEST_EPOCH}.pt
    PATH_OUT=$PATH_CPT/ABX/${BEST_EPOCH}/${TEST_LANGUAGE}/1s
#    mkdir -p $PATH_OUT
#    stool run /private/home/marvinlvn/InfTrain/CPC_torch/cpc/eval/eval_ABX.py --args="from_checkpoint $MODEL_PATH \
#                        $PATH_ITEM_FILE $ZEROSPEECH_DATASET --seq_norm --strict --file_extension $FILE_EXT --out $PATH_OUT" \
#                    --ngpu=1 --ncpu=8 \
#                    --name=InfTrain/$TEST_LANGUAGE/ABX/CPC_SMALL/family_$i \
#                    --partition=learnlab --anaconda=/private/home/marvinlvn/.conda/envs/inftrain
  done;
done;

# CPC big
#levels_gru=(1 2 3 4)
levels_gru=(4)
for i in {04..06}; do
  for TEST_LANGUAGE in ${TEST_LANGUAGES[*]}; do
    for level_gru in ${levels_gru[*]}; do
      PATH_ITEM_FILE=$ZR_PATH/${TEST_LANGUAGE}/1s/1s.item
      ZEROSPEECH_DATASET=$ZR_PATH/${TEST_LANGUAGE}/1s

      # CPC_torch same config than the one used on jean zay
      PATH_CPT=/checkpoint/marvinlvn/InfTrain/InfTrain_models/EN/50h/${i}/cpc_big
      BEST_EPOCH=$(python $HOME/CPC_audio_jzay/utils/best_val_epoch.py --model_path ${PATH_CPT} | grep -oP "(?<=is : )([0-9]+)")
      MODEL_PATH=${PATH_CPT}/checkpoint_${BEST_EPOCH}.pt
      PATH_OUT=$PATH_CPT/ABX/${BEST_EPOCH}/level_gru_${level_gru}/${TEST_LANGUAGE}/1s
      mkdir -p $PATH_OUT
      stool run /private/home/marvinlvn/InfTrain/CPC_torch/cpc/eval/eval_ABX.py --args="from_checkpoint $MODEL_PATH \
                          $PATH_ITEM_FILE $ZEROSPEECH_DATASET --seq_norm --strict --file_extension $FILE_EXT --out $PATH_OUT --level_gru ${level_gru}" \
                      --ngpu=1 --ncpu=8 \
                      --name=InfTrain/$TEST_LANGUAGE/ABX/CPC_BIG/family_$i/level_gru_${level_gru} \
                      --partition=learnlab --anaconda=/private/home/marvinlvn/.conda/envs/inftrain
    done;
  done;
done;
