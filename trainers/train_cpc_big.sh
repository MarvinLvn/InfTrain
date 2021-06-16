#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --partition=gpu_p2            # access to octo-gpus machines
#SBATCH --nodes=1                     # nombre de noeud
#SBATCH --gres=gpu:8                  # nombre de GPUs par nœud
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --exclusive


PATH_DB=$1
PATH_CPT=deduce from path db

python CPC_audio/cpc/train.py --pathCheckpoint ${PATH_CPT} \
                           --pathDB ${PATH_DB} --max_size_loaded 400000000 \
                           --file_extension .wav --nLevelsGRU 4 --hiddenEncoder 512 --hiddenGar 512 --save_step 1