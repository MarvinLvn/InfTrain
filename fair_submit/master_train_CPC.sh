#!/bin/bash

shopt -s expand_aliases
alias stool='/private/home/mriviere/FairInternal/stool/stool.py'

PATH_DB=/private/home/marvinlvn/DATA/CPC_data/train/InfTrain/200h/00
NB_EPOCHS=100

for i in $(seq 0 4); do

  PATH_CPT=/checkpoint/marvinlvn/InfTrain/200h/seed_${i}/cpc_torch_classic
  # CPC_torch same config than the one used on jean zay
  mkdir -p $PATH_CPT
  stool run /private/home/marvinlvn/InfTrain/CPC_torch/cpc/train.py --args="--pathCheckpoint ${PATH_CPT} \
                           --pathDB ${PATH_DB} --restart \
                           --file_extension .wav --nLevelsGRU 2 --save_step 2 --multihead_rnn --restart \
                           --nEpoch ${NB_EPOCHS} --random_seed $i --n_process_loader 1 --save_step 5 --ignore_cache" \
                  --ngpu=8 --ncpu=64 \
                  --name=InfTrain/$DATASET/cpc_torch_classic/seed_$i \
                  --partition=learnfair --anaconda=/private/home/marvinlvn/.conda/envs/inftrain

  # CPC_torch same config + schedulerRamp
  PATH_CPT=/checkpoint/marvinlvn/InfTrain/200h/seed_${i}/cpc_torch_scheduler_ramp
  mkdir -p $PATH_CPT
  stool run /private/home/marvinlvn/InfTrain/CPC_torch/cpc/train.py --args="--pathCheckpoint ${PATH_CPT} \
                           --pathDB ${PATH_DB} --restart \
                           --file_extension .wav --nLevelsGRU 2 --save_step 2 --multihead_rnn --restart \
                           --nEpoch ${NB_EPOCHS} --random_seed $i --n_process_loader 1 --save_step 5 --schedulerRamp=10  --ignore_cache" \
                  --ngpu=8 --ncpu=64 \
                  --name=InfTrain/$DATASET/cpc_torch_scheduler_ramp/seed_$i \
                  --partition=learnfair --anaconda=/private/home/marvinlvn/.conda/envs/inftrain

  # CPC2 no data aug, same speaker sampling
  PATH_CPT=/checkpoint/marvinlvn/InfTrain/200h/seed_${i}/cpc2_classic
  mkdir -p $PATH_CPT
  stool run /private/home/marvinlvn/CPC_audio_jzay/CPC_audio/cpc/train.py --args="--pathCheckpoint ${PATH_CPT} \
                           --pathDB ${PATH_DB} --restart \
                           --file_extension .wav --nLevelsGRU 2 --save_step 2 --multihead_rnn --restart \
                           --nEpoch ${NB_EPOCHS} --random_seed  $i --n_process_loader 1 --save_step 5 --samplingType samespeaker --ignore_cache" \
                  --ngpu=8 --ncpu=64 \
                  --name=InfTrain/$DATASET/cpc2_classic/seed_$i \
                  --partition=learnfair --anaconda=/private/home/marvinlvn/.conda/envs/cpc2

  # CPC2 data aug, temporal speaker sampling
  PATH_CPT=/checkpoint/marvinlvn/InfTrain/200h/seed_${i}/cpc2_data_aug_temporal
  mkdir -p $PATH_CPT
  stool run /private/home/marvinlvn/CPC_audio_jzay/CPC_audio/cpc/train.py --args="--pathCheckpoint ${PATH_CPT} \
                           --pathDB ${PATH_DB} --restart \
                           --file_extension .wav --nLevelsGRU 2 --save_step 2 --multihead_rnn --restart \
                           --nEpoch ${NB_EPOCHS} --random_seed $i --n_process_loader 1 --save_step 5 --samplingType temporalsamespeaker \
                           --augment_past --no_artefacts --augment_type pitch artificial_reverb --naming_convention=spkr_id_nb --ignore_cache" \
                  --ngpu=8 --ncpu=64 \
                  --name=InfTrain/$DATASET/cpc2_data_aug_temporal/seed_$i \
                  --partition=learnfair --anaconda=/private/home/marvinlvn/.conda/envs/cpc2

done;