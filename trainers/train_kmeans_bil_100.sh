#!/usr/bin/env bash
#SBATCH --account=cfs@gpu
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1                # nombre de GPU a reserver
#SBATCH --cpus-per-task=10          # nombre de cpus par GPU : rule of thumb n_cpu = 10*n_gpus
#SBATCH --hint=nomultithread

PATH_DB=$1

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters. Please respect :"
  echo "./train_cpc_kmeans_bil_100.sh /path/to/database/containing/wav/files"
  echo "Example :"
  echo "./train_cpc_kmeans_bil_100.sh /gpfsscratch/rech/cfs/commun/families/EN/50h/00"
  exit
fi

SHARE=$(basename $PATH_DB)
SIZE=$(basename $(dirname $PATH_DB))
LANGUAGE=$(basename $(dirname $(dirname $PATH_DB)))

CPC_NAME=cpc_small
NB_EPOCHS=300

PATH_CPT=/gpfsssd/scratch/rech/ank/ucv88ce/projects/MultilingualCPC/checkpoints/inftrain/${LANGUAGE}/${SIZE}/${SHARE}/kmeans100
PATH_CPC=/gpfsssd/scratch/rech/ank/ucv88ce/projects/MultilingualCPC/checkpoints/inftrain/${LANGUAGE}/${SIZE}/${SHARE}/${CPC_NAME}
BEST_EPOCH=$(python /gpfsdswork/projects/rech/ank/ucv88ce/projects/MultilingualCPC/utils/best_val_epoch.py --model_path ${PATH_CPC} | grep -oP "(?<=is : )([0-9]+)")
PATH_CPC=${PATH_CPC}/checkpoint_${BEST_EPOCH}.pt

LEVEL_GRU=2

if [ -f ${PATH_CPT}/running.state ]; then
  echo "${PATH_CPT}/running.state found. Not running anything."
  exit
fi;


#mkdir -p ${PATH_CPT}
#touch ${PATH_CPT}/running.state
echo "Start training $PATH_CPT"

module load sox
source activate inftrain
HOME=/gpfsdswork/projects/rech/ank/ucv88ce/
export PYTHONPATH=$HOME/repos/CPC_torch:$HOME/projects/MultilingualCPC/WavAugment:$PYTHONPATH

if [ ! -f ${PATH_CPT}/checkpoint_last.pt ]; then
    python $HOME/repos/CPC_torch/cpc/clustering/clustering_script.py ${PATH_CPC} ${PATH_CPT} ${PATH_DB} --recursionLevel 2 --extension wav \
           --nClusters 100 --MAX_ITER $NB_EPOCHS --save --batchSizeGPU 200 --level_gru ${LEVEL_GRU} --perIterSize 1406 --save-last 5 ;
fi

rm ${PATH_CPT}/running.state
if [ -f ${PATH_CPT}/checkpoint${NB_EPOCHS}.pt ]; then
  touch ${PATH_CPT}/done.state
fi;
