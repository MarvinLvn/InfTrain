#!/usr/bin/env bash

#------------------------ MONO ---------------------------------------------
lang=fr
PATH_AUDIO_FILES=/gpfsssd/scratch/rech/cfs/commun/families/EN/200h/01 #needs to be an absolute path
PATH_CHECKPOINT_DIR=experiments/checkpoints/inftrain/EN/200h/01
EXTENSION=".wav" 
NB_EPOCHS=100 # originally 200
#-----------------------------------------------------------------------


# #------------------------ MIX ---------------------------------------------
# set_l=A
# PATH_AUDIO_FILES=/gpfsssd/scratch/rech/ank/ucv88ce/data/cv-corpus-6.1-2020-12-11/wav_all #needs to be an absolute path
# PATH_CHECKPOINT_DIR=experiments/checkpoints/cv/mix/en-fr_$set_l
# PATH_TRAIN=data/cv/mix/en-fr/${set_l}_en-fr_train-200h-5000spk.txt
# PATH_VAL=data/cv/mix/en-fr/${set_l}_en-fr_val-10h-2000spk.txt
# EXTENSION=".wav" 
# NB_EPOCHS=100 # originally 200
# #-----------------------------------------------------------------------

mkdir -p ${PATH_CHECKPOINT_DIR}

sbatch -o ${PATH_CHECKPOINT_DIR}/train_CPC_small.txt ./trainers/train_cpc_small_inftrain.sh $PATH_AUDIO_FILES $PATH_CHECKPOINT_DIR $PATH_TRAIN $PATH_VAL $NB_EPOCHS
