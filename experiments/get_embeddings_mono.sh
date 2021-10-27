#!/bin/bash
#SBATCH --account=ank@gpu
#SBATCH --output=logs/get_embeddings_%A_%a.out
#SBATCH --nodes=1                     # nombre de noeud
#SBATCH --gres=gpu:1                  # nombre de GPUs par nœud
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --cpus-per-task=10

module load sox
source activate inftrain


HOURS="100h"

#--------------------------------------------------------------

for FAMILY_ID in "EN/${HOURS}/00" "FR/${HOURS}/00"; do
    MODEL_LOCATION="/gpfsssd/scratch/rech/ank/ucv88ce/projects/MultilingualCPC/checkpoints/inftrain/"
    BEST_EPOCH_PY="/gpfsdswork/projects/rech/ank/ucv88ce/projects/MultilingualCPC/utils/best_val_epoch.py"
    CHECKPOINT_LOCATION="${MODEL_LOCATION}${FAMILY_ID}"
    BEST_EPOCH="$(python "$BEST_EPOCH_PY" --output-id --model_path "${CHECKPOINT_LOCATION}/cpc_small")"
    CPC_CHECKPOINT_FILE="${MODEL_LOCATION}${FAMILY_ID}/cpc_small/checkpoint_${BEST_EPOCH}.pt"

    for lang in "en" "fr"; do
        python /gpfsdswork/projects/rech/ank/ucv88ce/repos/zerospeech2021_baseline/scripts/build_CPC_features.py \
               $CPC_CHECKPOINT_FILE \
               /gpfsscratch/rech/cfs/commun/cv21_ABX/raw_dataset/$lang \
               /gpfsssd/scratch/rech/ank/ucv88ce/projects/MultilingualCPC/eval/inftrain/${FAMILY_ID}/features/cpc_small/CV_$lang \
               --file_extension wav --gru_level 2 ;
    done;
done
