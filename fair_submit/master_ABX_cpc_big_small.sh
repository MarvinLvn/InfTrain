#!/bin/bash

# Marvin : you should check if that script is working.
# Thanks :)
shopt -s expand_aliases
alias stool='/private/home/mriviere/FairInternal/stool/stool.py'

ZR_PATH=/private/home/marvinlvn/DATA/CPC_data/test/zerospeech2017/data/test
TEST_LANGUAGES=(english french)
FILE_EXT=.wav

for i in $(seq 0 4); do
  for TEST_LANGUAGE in ${TEST_LANGUAGES[*]}; do
    PATH_ITEM_FILE=$ZR_PATH/${TEST_LANGUAGE}/1s/1s.item
    ZEROSPEECH_DATASET=$ZR_PATH/${TEST_LANGUAGE}/1s

    # CPC_torch same config than the one used on jean zay
    PATH_CPT=/checkpoint/marvinlvn/InfTrain/200h/seed_${i}/cpc_torch_classic
    BEST_EPOCH=$(python $HOME/CPC_audio_jzay/utils/best_val_epoch.py --model_path ${PATH_CPT} | grep -oP "(?<=is : )([0-9]+)")
    MODEL_PATH=${PATH_CPT}/checkpoint_${BEST_EPOCH}.pt
    PATH_OUT=$PATH_CPT/ABX/${BEST_EPOCH}/${TEST_LANGUAGE}/1s
    mkdir -p $PATH_OUT
    stool run /private/home/marvinlvn/InfTrain/CPC_torch/cpc/eval/eval_ABX.py --args="from_checkpoint $MODEL_PATH \
                        $PATH_ITEM_FILE $ZEROSPEECH_DATASET --seq_norm --strict --file_extension $FILE_EXT --out $PATH_OUT" \
                    --ngpu=1 --ncpu=8 \
                    --name=InfTrain/$DATASET/ABX/cpc_torch_classic/seed_$i \
                    --partition=learnlab --anaconda=/private/home/marvinlvn/.conda/envs/inftrain

    # CPC_torch same config + schedulerRamp
    PATH_CPT=/checkpoint/marvinlvn/InfTrain/200h/seed_${i}/cpc_torch_scheduler_ramp
    BEST_EPOCH=$(python $HOME/CPC_audio_jzay/utils/best_val_epoch.py --model_path ${PATH_CPT} | grep -oP "(?<=is : )([0-9]+)")
    MODEL_PATH=${PATH_CPT}/checkpoint_${BEST_EPOCH}.pt
    PATH_OUT=$PATH_CPT/ABX/${BEST_EPOCH}/${TEST_LANGUAGE}/1s
    mkdir -p $PATH_OUT
    stool run /private/home/marvinlvn/InfTrain/CPC_torch/cpc/eval/eval_ABX.py --args="from_checkpoint $MODEL_PATH \
                        $PATH_ITEM_FILE $ZEROSPEECH_DATASET --seq_norm --strict --file_extension $FILE_EXT --out $PATH_OUT" \
                    --ngpu=1 --ncpu=8 \
                    --name=InfTrain/$DATASET/ABX/cpc_torch_scheduler_ramp/seed_$i \
                    --partition=learnlab --anaconda=/private/home/marvinlvn/.conda/envs/inftrain

    # CPC2 no data aug, same speaker sampling
    PATH_CPT=/checkpoint/marvinlvn/InfTrain/200h/seed_${i}/cpc2_classic
    BEST_EPOCH=$(python $HOME/CPC_audio_jzay/utils/best_val_epoch.py --model_path ${PATH_CPT} | grep -oP "(?<=is : )([0-9]+)")
    MODEL_PATH=${PATH_CPT}/checkpoint_${BEST_EPOCH}.pt
    PATH_OUT=$PATH_CPT/ABX/${BEST_EPOCH}/${TEST_LANGUAGE}/1s
    mkdir -p $PATH_OUT
    stool run /private/home/marvinlvn/CPC_audio_jzay/CPC_audio/cpc/eval/ABX.py --args="from_checkpoint $MODEL_PATH \
                        $PATH_ITEM_FILE $ZEROSPEECH_DATASET --seq_norm --strict --file_extension $FILE_EXT --out $PATH_OUT" \
                    --ngpu=1 --ncpu=8 \
                    --name=InfTrain/$DATASET/cpc2_classic/seed_$i \
                    --partition=learnlab --anaconda=/private/home/marvinlvn/.conda/envs/cpc2

    # CPC2 data aug, temporal speaker sampling
    PATH_CPT=/checkpoint/marvinlvn/InfTrain/200h/seed_${i}/cpc2_data_aug_temporal
    BEST_EPOCH=$(python $HOME/CPC_audio_jzay/utils/best_val_epoch.py --model_path ${PATH_CPT} | grep -oP "(?<=is : )([0-9]+)")
    MODEL_PATH=${PATH_CPT}/checkpoint_${BEST_EPOCH}.pt
    PATH_OUT=$PATH_CPT/ABX/${BEST_EPOCH}/${TEST_LANGUAGE}/1s
    mkdir -p $PATH_OUT
    stool run /private/home/marvinlvn/CPC_audio_jzay/CPC_audio/cpc/eval/ABX.py --args="from_checkpoint $MODEL_PATH \
                        $PATH_ITEM_FILE $ZEROSPEECH_DATASET --seq_norm --strict --file_extension $FILE_EXT --out $PATH_OUT" \
                    --ngpu=1 --ncpu=8 \
                    --name=InfTrain/$DATASET/cpc2_data_aug_temporal/seed_$i \
                    --partition=learnlab --anaconda=/private/home/marvinlvn/.conda/envs/cpc2
  done;
done;