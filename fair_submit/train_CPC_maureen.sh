#!/bin/bash

shopt -s expand_aliases
alias stool='/private/home/mriviere/FairInternal/stool/stool.py'

PATH_DB=/private/home/marvinlvn/DATA/CPC_data/train/CommonVoicesMaureen/wav_all
PATH_TRAIN=/private/home/marvinlvn/DATA/CPC_data/train/CommonVoicesMaureen/splits/mono/en/train-200h-5000spk.txt
PATH_VAL=/private/home/marvinlvn/DATA/CPC_data/train/CommonVoicesMaureen/splits/mono/en/val-10h-2000spk.txt

PATH_CPT=/checkpoint/marvinlvn/InfTrain/maureen/cv_wav_all_en_train_val_specified
mkdir -p $PATH_CPT
stool run /private/home/marvinlvn/CPC_audio_jzay/CPC_audio/cpc/train.py --args="--pathCheckpoint ${PATH_CPT} \
                           --pathDB ${PATH_DB} --pathTrain ${PATH_TRAIN} --pathVal ${PATH_VAL}  \
                           --file_extension .wav --nLevelsGRU 2 --multihead_rnn --no_artefacts --augment_past --augment_type pitch artificial_reverb \
                           --nEpoch 100 --n_process_loader 1 --save_step 5 --samplingType samespeaker --ignore_cache" \
                  --ngpu=8 --ncpu=64 \
                  --name=InfTrain/maureen/cpc2_maureen_train_val_specified \
                  --partition=learnfair --anaconda=/private/home/marvinlvn/.conda/envs/cpc2