#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=10:00:00

##
## Usage: ./evaluate_CPC_loss.sh /path/to/duration/family_id
##
## Example:
##
## ./evaluate_CPC_loss.sh /gpfsscratch/rech/cfs/commun/families/FR/50h/11
##
## Parameters:
##
##   FAMILY_PATH (ID)            the path to the family that we want to evaluate.
##
## ENVIRONMENT VARIABLES :
##
## MODEL_LOCATION                the root directory containing all the model checkpoint files (default: /gpfsscratch/rech/cfs/commun/InfTrain_models)
## FAMILIES_LOCATION             the root directory containing source dataset with families (default: /gpfsscratch/rech/cfs/commun/families)
## ZEROSPEECH_DATASET            the location of the common voice dataset used for evaluation (default: /gpfsscratch/rech/cfs/commun/marvin_eval/ABX_CV)
## FILE_EXTENSION                the extension to use as input in the feature extraction (default: wav)
## EVAL_NB_JOBS                  the number of jobs to use for evaluation (default: 20)
## BEST_EPOCH_PY                 the script to use to find the best epoch checkpoint (default: /gpfsscratch/rech/cfs/uow84uh/InfTrain/utils/best_val_epoch.py)
##
## More info:
## https://github.com/MarvinLvn/CPC_torch
## https://docs.google.com/spreadsheets/d/1pcT_6YLdQ5Oa2pO21mRKzzU79ZPUZ-BfU2kkXg2mayE/edit?usp=drive_web&ouid=112305914309228781110

# check only parameters without running eval
DRY_RUN="${DRY_RUN:-false}"

# --- Various utility functions & variables

# absolute path to the directory where this script is
here="$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null 2>&1 && pwd)"

# grep double comment lines for usage
function usage() {
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

MODEL_LOCATION="${MODEL_LOCATION:-/gpfsscratch/rech/cfs/commun/InfTrain_models}"
FAMILIES_LOCATION="${FAMILIES_LOCATION:-/gpfsscratch/rech/cfs/commun/families}"
ZEROSPEECH_DATASET="${ZEROSPEECH_DATASET:-/gpfsscratch/rech/cfs/commun/marvin_eval/ABX_CV}"
BEST_EPOCH_PY="${BEST_EPOCH_PY:-/gpfsscratch/rech/cfs/uow84uh/InfTrain/utils/best_val_epoch.py}"
LOSS_PY="${LOSS_PY:-/gpfsscratch/rech/cfs/uow84uh/InfTrain/utils/validate_CPC.py}"
FILE_EXT="${FILE_EXTENSION:-.wav}"
NB_JOBS="${EVAL_NB_JOBS:-20}"

PATH_TO_FAMILY=$1
shift

# check scripts locations
[ ! -f "$BEST_EPOCH_PY" ] && die "utils/best_val_epoch.py script was not found here : $BEST_EPOCH_PY"
[ ! -f "${LOSS_PY}" ] && die "Loss script was not found at : ${LOSS_PY}"


FAMILY_ID="${PATH_TO_FAMILY#${FAMILIES_LOCATION}}"
CHECKPOINT_LOCATION="${MODEL_LOCATION}${FAMILY_ID}"
CPC_CHECKPOINT_FILE=""
OUTPUT_LOCATION="${CHECKPOINT_LOCATION}"

# Find best epoch checkpoint to use for evaluation
if [ -d "${CHECKPOINT_LOCATION}/cpc_small" ]; then
  BEST_EPOCH="$(python "$BEST_EPOCH_PY" --output-id --model_path "${CHECKPOINT_LOCATION}/cpc_small")"
  CPC_CHECKPOINT_FILE="${MODEL_LOCATION}${FAMILY_ID}/cpc_small/checkpoint_${BEST_EPOCH}.pt"
  OUTPUT_LOCATION="${CHECKPOINT_LOCATION}/cpc_small/scores/CommonVoiceLoss"
elif [ -d "${CHECKPOINT_LOCATION}/cpc_big" ]; then
  BEST_EPOCH="$(python "$BEST_EPOCH_PY" --output-id --model_path "${CHECKPOINT_LOCATION}/cpc_big")"
  CPC_CHECKPOINT_FILE="${MODEL_LOCATION}${FAMILY_ID}/cpc_big/checkpoint_${BEST_EPOCH}.pt"
  OUTPUT_LOCATION="${CHECKPOINT_LOCATION}/cpc_big/scores/CommonVoiceLoss"
else
  die "No CPC checkpoints found for family ${FAMILY_ID}"
fi

# Verify INPUTS
[ ! -d $ZEROSPEECH_DATASET ] && die "ZEROSPEECH_DATASET not found: $ZEROSPEECH_DATASET"
[ ! -d $MODEL_LOCATION ] && die "Model location does not exist: $MODEL_LOCATION"
[ ! -d $FAMILIES_LOCATION ] && die "Families location does not exist: $FAMILIES_LOCATION"
[ ! -d $PATH_TO_FAMILY ] && die "Given family was not found: $PATH_TO_FAMILY"


#--debug print values && exit
if [[ $DRY_RUN == "true" ]]; then
  echo "family-id: $FAMILY_ID"
  echo "zerospeech-dataset: $ZEROSPEECH_DATASET"
  echo "model-location: $MODEL_LOCATION"
  echo "families-location: $FAMILIES_LOCATION"
  echo "scripts: (loss;$LOSS_PY) (epoch;$BEST_EPOCH_PY)"
  echo "checkpoint-file: $CPC_CHECKPOINT_FILE"
  echo "output-location: $OUTPUT_LOCATION"
  echo "file-extension: $FILE_EXT"
  echo "nb-jobs: $NB_JOBS"
  echo "python $(which python)"
  for lang in fr en; do
    PATH_OUT="$OUTPUT_LOCATION/${lang}/${BEST_EPOCH}/loss.json"
    echo "==> python $LOSS_PY --pathModel $CPC_CHECKPOINT_FILE --pathOut $PATH_OUT --pathDB $ZEROSPEECH_DATASET/${lang} --file_extension $FILE_EXT"
  done;
  exit 0
fi

for lang in fr en; do
  LANG_DATASET="${ZEROSPEECH_DATASET}/${lang}"
  PATH_OUT="$OUTPUT_LOCATION/${lang}/${BEST_EPOCH}/loss.json"
  mkdir -p "$(dirname PATH_OUT)"
  srun python $LOSS_PY --pathModel $CPC_CHECKPOINT_FILE --pathOut $PATH_OUT --pathDB $LANG_DATASET --file_extension $FILE_EXT
done
