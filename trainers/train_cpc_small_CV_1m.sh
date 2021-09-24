#!/bin/bash
#SBATCH --account=ank@gpu
#SBATCH --partition=gpu_p2            # access to octo-gpus machines
#SBATCH --nodes=1                     # nombre de noeud
#SBATCH --gres=gpu:8                  # nombre de GPUs par nœud
#SBATCH --time=60:00:00               #--time=60:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --exclusive
#SBATCH --cpus-per-task=3             #MDS augment memory
#SBATCH --qos=qos_gpu-t4    #run --qos=qos_gpu-t4

#export MASTER=`hostname`
#export MASTER_PORT=13369
#export NCCL_DEBUG=INFO # for debugging


# Across-machines training
export MASTER=`hostname`
export MASTER_PORT=13369
export NCCL_DEBUG=INFO # for debugging

PATH_DB=$1
PATH_CPT=$2
PATH_TRAIN=$3
PATH_VAL=$4
NB_EPOCHS=$5

if [ "$#" -ne 5 ]; then
  echo "Illegal number of parameters. Please respect :"
  echo "./train_cpc_small.sh /path/to/database/containing/wav/files"
  echo "Example :"
  echo "./train_cpc_small.sh /gpfsscratch/rech/cfs/commun/families/EN/50h/00"
  exit
fi


if [ -f ${PATH_CPT}/running.state ]; then
  echo "${PATH_CPT}/running.state found. Not running anything."
fi;

mkdir -p $PATH_CPT
touch ${PATH_CPT}/running.state

echo "Start training $PATH_CPT"
module load sox
# source activate cpc2
#source activate multicpc
source activate inftrain
HOME=/gpfsdswork/projects/rech/ank/ucv88ce/
export PYTHONPATH=$HOME/repos/CPC_torch:$HOME/projects/MultilingualCPC/WavAugment:$PYTHONPATH

#removed "restart"

max_size_loaded=4000000 #divide by 100 to fit in memory
python /gpfsdswork/projects/rech/ank/ucv88ce/repos/CPC_torch/cpc/train.py --pathCheckpoint ${PATH_CPT} \
     --pathDB ${PATH_DB} \
     --pathTrain ${PATH_TRAIN} --pathVal ${PATH_VAL} \
                           --file_extension .wav --nLevelsGRU 2 --save_step 2 --multihead_rnn \
                           --nEpoch ${NB_EPOCHS} --random_seed 42 \
                           --n_process_loader 1 \
                           --max_size_loaded ${max_size_loaded} \
                           --schedulerRamp 10


rm ${PATH_CPT}/running.state
if [ -f ${PATH_CPT}/checkpoint_$(($NB_EPOCHS-1)).pt ]; then
  touch ${PATH_CPT}/done.state
fi;