#!/bin/bash
#SBATCH --account=ank@gpu
#SBATCH --nodes=2                     # nombre de noeud
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4                  # nombre de GPUs par nœud
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --exclusive

# Across-machines training
export MASTER=`hostname`
export MASTER_PORT=13369
export NCCL_DEBUG=INFO # for debugging

PATH_DB=$1

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters. Please respect :"
  echo "./train_cpc_small.sh /path/to/database/containing/wav/files"
  echo "Example :"
  echo "./train_cpc_small.sh /gpfsscratch/rech/cfs/commun/families/EN/50h/00"
  exit
fi

SHARE=$(basename $PATH_DB)
SIZE=$(basename $(dirname $PATH_DB))
LANGUAGE=$(basename $(dirname $(dirname $PATH_DB)))
PATH_CPT=/gpfsssd/scratch/rech/ank/ucv88ce/projects/MultilingualCPC/checkpoints/inftrain/${LANGUAGE}/${SIZE}/${SHARE}/cpc_small

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
  NB_EPOCHS=50;
else
  echo "Not possible to deduce the number of epochs from the size of the training set."
  echo "You should check that you haven't called train_cpc_small.sh with a training set whose size is greater or equal than 800h"
  exit
fi;

if [ -f ${PATH_CPT}/running.state ]; then
  echo "${PATH_CPT}/running.state found. Not running anything."
fi;

if [ -f ${PATH_CPT}/done.state ]; then
    echo "${PATH_CPT}/done.state found. Not running anything."
    exit 1
fi;


mkdir -p $PATH_CPT
touch ${PATH_CPT}/running.state

echo "Start training $PATH_CPT"
module load sox
source activate inftrain

HOME=/gpfsdswork/projects/rech/ank/ucv88ce/
export PYTHONPATH=$HOME/repos/CPC_torch:$HOME/projects/MultilingualCPC/WavAugment:$PYTHONPATH


srun python /gpfsdswork/projects/rech/ank/ucv88ce/repos/CPC_torch/cpc/train.py --pathCheckpoint ${PATH_CPT} \
                           --pathDB ${PATH_DB} \
                           --file_extension .wav --nLevelsGRU 2 --multihead_rnn \
                           --nEpoch ${NB_EPOCHS} --random_seed 42 --n_process_loader 1 --save_step 5 \
                           --distributed --master_port $MASTER_PORT --schedulerRamp 10


rm ${PATH_CPT}/running.state
if [ -f ${PATH_CPT}/checkpoint_$(($NB_EPOCHS-1)).pt ]; then
  touch ${PATH_CPT}/done.state
fi;
