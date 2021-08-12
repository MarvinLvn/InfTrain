#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --nodes=1                     # nombre de noeud
#SBATCH --gres=gpu:1                  # nombre de GPUs par nœud
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --cpus-per-task=10
##
## Usage: ./evaluate_kmeans.sh /path/to/duration/family_id
##
## 1) quantize_audio
## 2) build_1hot
## 3) python eval/eval_ABX.py from_pre_computed ${PATH_FEATURE_DIR} $ITEM_PATH --file_extension .pt --out $OUTPUT_DIR --feature_size $FEAT_SIZE  --not_normalize
##
## Example:
##
## ./evaluate_kmeans.sh /gpfsscratch/rech/cfs/commun/families/FR/50h/11
##
## Parameters:
##
##   FAMILY_PATH (ID)            the path to the family that we want to evaluate.
##
## ENVIRONMENT VARIABLES :
##
## MODEL_LOCATION                the root directory containing all the model checkpoint files (default: /gpfsscratch/rech/cfs/commun/InfTrain_models)
## FAMILIES_LOCATION             the root directory containing source dataset with families (default: /gpfsscratch/rech/cfs/commun/families)
## ZEROSPEECH_DATASET            the location of the zerospeech dataset used for evaluation (default: /gpfsssd/scratch/rech/cfs/commun/zerospeech2017/data/test)
## FILE_EXTENSION                the extension to use as input in the feature extraction (default: wav)
## ABX_PY                        the script to use for abx evaluation (default: /gpfsscratch/rech/cfs/uow84uh/InfTrain/CPC_torch/cpc/eval/eval_ABX.py)
## BEST_EPOCH_PY                 the script to use to find the best epoch checkpoint (default: /gpfsscratch/rech/cfs/uow84uh/nick_temp/InfTrain/utils/best_val_epoch.py)
##
## More info:
## https://github.com/MarvinLvn/CPC_torch
## https://docs.google.com/spreadsheets/d/1pcT_6YLdQ5Oa2pO21mRKzzU79ZPUZ-BfU2kkXg2mayE/edit?usp=drive_web&ouid=112305914309228781110
## https://github.com/bootphon/zerospeech2021_baseline

# check only parameters without running eval
DRY_RUN="${DRY_RUN:-false}"
# clean up features after completion of evaluation
CLEAN_UP="${CLEAN_UP:-true}"

# --- Various utility functions & variables

# grep double comment lines for usage
function usage
{
    sed -nr 's/^## ?//p' ${BASH_SOURCE[0]}
    exit 0
}

# console messaging
msg() {
  echo >&2 -e "${1-}"
}

# error exit
function die() {
  local msg=$1
  local code=${2-1} # default exit status 1
  msg "$msg"
  exit "$code"
}

# --- Input Validation & Processing
# input arguments
[ "$1" == "-h" -o "$1" == "-help" -o "$1" == "--help" ] && usage
[ $# -lt 1 ] && usage

# Paths
MODEL_LOCATION="${MODEL_LOCATION:-/gpfsscratch/rech/cfs/commun/InfTrain_models}"
FAMILIES_LOCATION="${FAMILIES_LOCATION:-/gpfsscratch/rech/cfs/commun/families}"
ZEROSPEECH_DATASET="${ZEROSPEECH_DATASET:-/gpfsssd/scratch/rech/cfs/commun/zerospeech2017/data/test}"
FILE_EXT="${FILE_EXTENSION:-.wav}"
FEAT_SIZE="${FEAT_SIZE:-0.01}"

# Scripts
ABX_PY="${ABX_PY:-/gpfsscratch/rech/cfs/uow84uh/InfTrain/CPC_torch/cpc/eval/eval_ABX_clustering.py}"
BEST_EPOCH_PY="${BEST_EPOCH_PY:-/gpfsscratch/rech/cfs/uow84uh/nick_temp/InfTrain/utils/best_val_epoch.py}"

# Arguments
PATH_TO_FAMILY=$1
shift;

# check scripts locations
[ ! -f "$BEST_EPOCH_PY" ] && die "best_val_epoch.py script was not at : $BEST_EPOCH_PY"
[ ! -f "${ABX_PY}" ] && die "ABX script was not found at : ${ABX_PY}"

FAMILY_ID="${PATH_TO_FAMILY#${FAMILIES_LOCATION}}"
CHECKPOINT_LOCATION="${MODEL_LOCATION}${FAMILY_ID}"
CHECKPOINT_FILE=""
OUTPUT_LOCATION=""

# Find best epoch checkpoint to use for evaluation
#if [ -d "${CHECKPOINT_LOCATION}/kmeans_50" ]; then
if [ -d "${CHECKPOINT_LOCATION}/bad_folder" ]; then
  BEST_EPOCH="$(python "$BEST_EPOCH_PY" --output-id --model_path "${CHECKPOINT_LOCATION}/kmeans_50")"
  CHECKPOINT_FILE="${MODEL_LOCATION}${FAMILY_ID}/kmeans_50/checkpoint_${BEST_EPOCH}.pt"
  OUTPUT_LOCATION="${CHECKPOINT_LOCATION}/kmeans_50/ABX/${BEST_EPOCH}"
else
#  die "No CPC-kmeans checkpoints found for family ${FAMILY_ID}"
  echo "debugging: remove this !!!"
  CHECKPOINT_FILE="/gpfsstore/rech/cfs/commun/zr2021_models/checkpoints/CPC-small-kmeans50/clustering_kmeans50/clustering_CPC_small_kmeans50.pt"
  OUTPUT_LOCATION="${CHECKPOINT_LOCATION}/kmeans_50/ABX/50"
fi

# Verify INPUTS
[ ! -d $ZEROSPEECH_DATASET ] && die "ZEROSPEECH_DATASET not found: $ZEROSPEECH_DATASET"
[ ! -d $MODEL_LOCATION ] && die "Model location does not exist: $MODEL_LOCATION"
[ ! -d $FAMILIES_LOCATION ] && die "Families location does not exist: $FAMILIES_LOCATION"
[ ! -d $PATH_TO_FAMILY ] && die "Given family was not found: $PATH_TO_FAMILY"


#--debug print values && exit
if [[ $DRY_RUN == "true" ]]; then
  echo "-------------- VARIABLES ---------------------------"
  echo "family-id: $FAMILY_ID"
  echo "features-location: $FEATURES_LOCATION"
  echo "zerospeech-dataset: $ZEROSPEECH_DATASET"
  echo "model-location: $MODEL_LOCATION"
  echo "families-location: $FAMILIES_LOCATION"
  echo "scripts: (abx;$ABX_PY), (epoch;$BEST_EPOCH_PY)"
  echo "checkpoint-file: $CHECKPOINT_FILE"
  echo "output-location: $OUTPUT_LOCATION"
  echo "file-extension: $FILE_EXT"
  echo "python $(which python)"
  echo "-------------- CMDs ---------------------------"
fi

for lang in french english
do
  DATA="${ZEROSPEECH_DATASET}/${lang}/1s"
  ITEM_PATH="${DATA}/${lang}/1s/1s.item"

  PATH_OUT="$OUTPUT_LOCATION/${lang}"
  OUT_FILE="${PATH_OUT}/abx_scores.json"

  if [[ $DRY_RUN == "true" ]]; then
    echo "=> python $ABX_PY --file-extension .wav --name-output $OUT_FILE --path_audio_data $DATA --path_abx_item $ITEM_PATH --clustering $CHECKPOINT_FILE"
    echo "-----------------------------"
  else
    mkdir -p $PATH_OUT
    srun python $ABX_PY --file-extension .wav --name-output $OUT_FILE --path_audio_data $DATA --path_abx_item $ITEM_PATH --clustering $CHECKPOINT_FILE --debug
  fi
done

echo "evaluation of $FAMILY_ID complete. ABX scores can be found @ $OUTPUT_LOCATION"
