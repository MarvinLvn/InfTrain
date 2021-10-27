#!/usr/bin/env bash
#SBATCH --account=ank@gpu
#SBATCH --mem=128G
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1                # nombre de GPU a reserver
#SBATCH --cpus-per-task=10          # nombre de cpus par GPU : rule of thumb n_cpu = 10*n_gpus
#SBATCH --qos=qos_gpu-t4
#SBATCH --hint=nomultithread

PATH_DB=$1
N_CLUSTER=$2

if [ "$#" -ne 2 ]; then
  echo "Illegal number of parameters. Please respect :"
  echo "./train_cpc_kmeans.sh /path/to/database/containing/wav/files N_CLUSTER"
  echo "Example :"
  echo "./train_cpc_kmeans.sh /gpfsscratch/rech/cfs/commun/families/EN/50h/00 50"
  exit
fi

SHARE=$(basename $PATH_DB)
SIZE=$(basename $(dirname $PATH_DB))
LANGUAGE=$(basename $(dirname $(dirname $PATH_DB)))

CPC_NAME=cpc_small
NB_EPOCHS=300
ALL_CCFRSCRATCH=/gpfsscratch/rech/cfs/commun/
PATH_CPT=${ALL_CCFRSCRATCH}/InfTrain_models/${LANGUAGE}/${SIZE}/${SHARE}/kmeans${N_CLUSTER}
PATH_CPC=${ALL_CCFRSCRATCH}/InfTrain_models/${LANGUAGE}/${SIZE}/${SHARE}/${CPC_NAME}
BEST_EPOCH=$(python /gpfsscratch/rech/cfs/uow84uh/InfTrain/utils/best_val_epoch.py --model_path ${PATH_CPC} | grep -oP "(?<=is : )([0-9]+)")
PATH_CPC=${PATH_CPC}/checkpoint_${BEST_EPOCH}.pt

LEVEL_GRU=2

module load sox
source activate inftrain
HOME=/gpfsdswork/projects/rech/ank/ucv88ce/
export PYTHONPATH=$HOME/repos/CPC_torch:$HOME/projects/MultilingualCPC/WavAugment:$PYTHONPATH

#mkdir -p ${PATH_CPT}
#touch ${PATH_CPT}/running.state
echo "Start training $PATH_CPT"

echo "Loading Prev CP"
python $HOME/repos/CPC_torch/cpc/clustering/clustering_script.py ${PATH_CPC} ${PATH_CPT} ${PATH_DB} --recursionLevel 2 --extension wav --nClusters $N_CLUSTER --MAX_ITER $NB_EPOCHS --save --batchSizeGPU 200 --level_gru ${LEVEL_GRU} --perIterSize 1406 --save-last 5

rm ${PATH_CPT}/running.state
if [ -f ${PATH_CPT}/checkpoint${NB_EPOCHS}.pt ]; then
  touch ${PATH_CPT}/done.state
fi;
