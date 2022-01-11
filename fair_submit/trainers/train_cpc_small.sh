#!/bin/bash
#SBATCH --partition=devlab
#SBATCH --nodes=2                     # nombre de noeud
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4                  # nombre de GPUs par nœud
#SBATCH --mem=512G
#SBATCH --time=72:00:00

# Across-machines training
export MASTER=`hostname`
export MASTER_PORT=$((13001 + $RANDOM % 2000))

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

if [ "$SIZE" == "wav" ] && [ "$LANGUAGE" == "dataset" ]; then
  SHARE=00
  SIZE=full
  LANGUAGE=$(basename $PATH_DB)
fi;

PATH_CPT=/checkpoint/marvinlvn/InfTrain/${LANGUAGE}/${SIZE}/${SHARE}/cpc_small

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
  NB_EPOCHS=20;
fi;

mkdir -p $PATH_CPT
echo "Start training $PATH_CPT"
srun python /private/home/marvinlvn/InfTrain/CPC_torch/cpc/train.py  --pathCheckpoint ${PATH_CPT} \
                           --pathDB ${PATH_DB} \
                           --file_extension .wav --nLevelsGRU 2 --save_step 1 --multihead_rnn \
                           --nEpoch ${NB_EPOCHS} --random_seed 42 --n_process_loader 1 \
                           --distributed --max_size_loaded 200000000 --master_port $MASTER_PORT --schedulerRamp 10
exit