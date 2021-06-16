#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --partition=gpu_p2            # access to octo-gpus machines
#SBATCH --nodes=1                     # nombre de noeud
#SBATCH --gres=gpu:8                  # nombre de GPUs par nœud
#SBATCH --time=20:00:00
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