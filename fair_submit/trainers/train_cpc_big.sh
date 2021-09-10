#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --nodes=8                     # nombre de noeud
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # nombre de GPUs par nœud
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive

# Across-machines training
export MASTER=`hostname`
export MASTER_PORT=13369
export NCCL_DEBUG=INFO # for debugging

# Script parameters
PATH_DB=$1

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters. Please respect :"
  echo "./train_cpc_big.sh /path/to/database/containing/wav/files"
  echo "Example :"
  echo "./train_cpc_big.sh /gpfsscratch/rech/cfs/commun/families/EN/50h/00"
  exit
fi

SHARE=$(basename $PATH_DB)
SIZE=$(basename $(dirname $PATH_DB))
LANGUAGE=$(basename $(dirname $(dirname $PATH_DB)))
PATH_CPT=${ALL_CCFRSCRATCH}/InfTrain_models/${LANGUAGE}/${SIZE}/${SHARE}/cpc_big

# This can be changed if we observe that models take too much time to train.
# However, if one decide to lower the number of epochs,
# one should check that models are indeed converging
if [ "$SIZE" == "50h" ]; then
  NB_EPOCHS=200
elif [ "$SIZE" == "100h" ]; then
  # 100 epochs for a 100h training set seems fine to me
  NB_EPOCHS=100
elif [ "$SIZE" == "200h" ]; then
  NB_EPOCHS=80
elif [ "$SIZE" == "400h" ]; then
  NB_EPOCHS=60;
elif [ "$SIZE" == "800h" ]; then
  NB_EPOCHS=50;
elif [ "$SIZE" == "1600h" ]; then
  NB_EPOCHS=40;
elif [ "$SIZE" == "3200h" ]; then
  NB_EPOCHS=30;
else
  echo "Not possible to deduce the number of epochs from the size of the training set."
  echo "You should check that you haven't called train_cpc_small.sh with a training set whose size is greater or equal than 800h"
  exit
fi;

if [ -f ${PATH_CPT}/running.state ]; then
  echo "${PATH_CPT}/running.state found. Not running anything."
  exitmoud
fi;

mkdir -p $PATH_CPT
touch ${PATH_CPT}/running.state
echo "Start training $PATH_CPT"
module load sox
srun python /gpfsscratch/rech/cfs/uow84uh/InfTrain/CPC_torch/cpc/train.py --pathCheckpoint ${PATH_CPT} \
                           --pathDB ${PATH_DB} --max_size_loaded 200000000 \
                           --file_extension .wav --nLevelsGRU 4 --hiddenEncoder 512 --hiddenGar 512 --save_step 1 \
                           --multihead_rnn --nEpoch ${NB_EPOCHS} --random_seed 42 --n_process_loader 1 --dropout --schedulerRamp 10 \
                           --batchSizeGPU 16 --rnnMode transformer --distributed --master_port $MASTER_PORT

rm ${PATH_CPT}/running.state
if [ -f ${PATH_CPT}/checkpoint_$(($NB_EPOCHS-1)).pt ]; then
  touch ${PATH_CPT}/done.state
fi;
