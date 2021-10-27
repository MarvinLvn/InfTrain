#!/bin/bash
#SBATCH --account=ank@gpu
#SBATCH --nodes=1                     # nombre de noeud
#SBATCH --gres=gpu:1                  # nombre de GPUs par nœud
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --cpus-per-task=10
##
## Usage: ./evaluate_kmeans.sh /path/to/duration/family_id NCLUSTER
##
## 1) quantize_audio
## 2) build_1hot
## 3) python eval/eval_ABX.py from_pre_computed ${PATH_FEATURE_DIR} $ITEM_PATH --file_extension .pt --out $OUTPUT_DIR --feature_size $FEAT_SIZE  --not_normalize
##
## Example:
##
## ./evaluate_kmeans.sh /gpfsscratch/rech/cfs/commun/families/FR/50h/11 5°
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
MODEL_LOCATION="${MODEL_LOCATION:-/gpfsssd/scratch/rech/ank/ucv88ce/projects/MultilingualCPC/checkpoints/inftrain/}"
FAMILIES_LOCATION="${FAMILIES_LOCATION:-/gpfsssd/scratch/rech/cfs/commun/families/}"
CV_DATASET="${CV_DATASET:-/gpfsscratch/rech/cfs/commun/cv21_ABX/raw_dataset}"
FILE_EXT="${FILE_EXTENSION:-.wav}"
FEAT_SIZE="${FEAT_SIZE:-0.01}"

# Scripts
ABX_PY="${ABX_PY:-/gpfsdswork/projects/rech/ank/ucv88ce/repos/CPC_torch/cpc/eval/eval_ABX_clustering.py}"
BEST_EPOCH_PY="${BEST_EPOCH_PY:-/gpfsdswork/projects/rech/ank/ucv88ce/projects/MultilingualCPC/utils/best_val_epoch.py}"

# Arguments
PATH_TO_FAMILY=$1
NCLUSTER=$2
CONTRAST=$3
LANG=$4
shift;

# check scripts locations
[ ! -f "$BEST_EPOCH_PY" ] && die "best_val_epoch.py script was not at : $BEST_EPOCH_PY"
[ ! -f "${ABX_PY}" ] && die "ABX script was not found at : ${ABX_PY}"

FAMILY_ID="${PATH_TO_FAMILY:${#FAMILIES_LOCATION}}"
CHECKPOINT_LOCATION="${MODEL_LOCATION}${FAMILY_ID}"
CHECKPOINT_FILE=""
OUTPUT_LOCATION=""

# Find best epoch checkpoint to use for evaluation
if [ -d "${CHECKPOINT_LOCATION}/kmeans${NCLUSTER}" ]; then
#if [ -d "${CHECKPOINT_LOCATION}/bad_folder" ]; then
    CHECKPOINT_FILE="${MODEL_LOCATION}${FAMILY_ID}/kmeans${NCLUSTER}/checkpoint_last.pt"
    OUTPUT_LOCATION="${CHECKPOINT_LOCATION}/kmeans${NCLUSTER}/ABX_CV/last"
else
  die "No CPC-kmeans checkpoints found for family ${FAMILY_ID}"
fi

# Verify INPUTS
[ ! -d $CV_DATASET ] && die "CV_DATASET not found: $CV_DATASET"
[ ! -d $MODEL_LOCATION ] && die "Model location does not exist: $MODEL_LOCATION"
[ ! -d $FAMILIES_LOCATION ] && die "Families location does not exist: $FAMILIES_LOCATION"
[ ! -d $PATH_TO_FAMILY ] && die "Given family was not found: $PATH_TO_FAMILY"



module load sox
source activate inftrain

HOME=/gpfsdswork/projects/rech/ank/ucv88ce/
export PYTHONPATH=$HOME/repos/CPC_torch:$HOME/projects/MultilingualCPC/WavAugment:$PYTHONPATH


for lang in ${LANG};
do
  DATA="${CV_DATASET}/${lang}"
  ITEM_PATH="${DATA}/${CONTRAST}.item"

  PATH_OUT="$OUTPUT_LOCATION/${CONTRAST}"
  OUT_FILE="${PATH_OUT}/ABX_scores.json"

  if [[ $DRY_RUN == "true" ]]; then
    echo "=> python $ABX_PY --file-extension .wav --name-output $OUT_FILE --path_audio_data $DATA --path_abx_item $ITEM_PATH --clustering $CHECKPOINT_FILE"
    echo "-----------------------------"
  else
    mkdir -p $PATH_OUT
    srun python $ABX_PY --file-extension .wav --name-output $OUT_FILE --path_audio_data $DATA --path_abx_item $ITEM_PATH --clustering $CHECKPOINT_FILE
  fi
done

echo "evaluation of $FAMILY_ID complete. ABX scores can be found @ $OUTPUT_LOCATION"
