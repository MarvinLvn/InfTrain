#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --nodes=2                     # nombre de noeud
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:4                  # nombre de GPUs par nœud
#SBATCH --time=100:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --exclusive


PATH_DB=$1
SHARE=$(basename $(dirname $MODEL_PATH))
SIZE=$(basename $(dirname $(dirname $MODEL_PATH)))
LANGUAGE=$(basename $(dirname $(dirname $(dirname $MODEL_PATH))))
PATH_CPT=${ALL_CCFRSCRATCH}/${LANGUAGE}/${SIZE}/${SHARE}
NB_EPOCHS=deduce from path db

if [ -f ${PATH_CPT}/running.state ]; then
  echo "${PATH_CPT}/running.state found. Not running anything."
fi;

touch ${PATH_CPT}/running.state
python CPC_audio/cpc/train.py --pathCheckpoint ${PATH_CPT} \
                           --pathDB ${PATH_DB} \
                           --file_extension .wav --nLevelsGRU 2 --save_step 2 --restart

rm ${PATH_CPT}/running.state
if [ -f ${PATH_CPT}/checkpoint${NB_EPOCHS}.pt ]; then
  touch ${PATH_CPT}/done.state
fi;