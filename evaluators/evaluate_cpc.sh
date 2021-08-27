#!/bin/bash
#SBATCH --account=ank@gpu
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=experiments/logs/eval_cpc_%j.out
##
## Usage: ./evaluate_cpc.sh /path/to/duration/family_id
##
## 1) Extract CPC representations on zerospeech2017/{french, english}
## 2) Compute ABX error rate
## This is done using abx evaluation in CPC_torch package
##
## Example:
##
## ./evaluate_cpc.sh /gpfsscratch/rech/cfs/commun/families/FR/50h/11
##
## Parameters:
##
##   FAMILY_PATH (ID)            the path to the family that we want to evaluate.
##
## ENVIRONMENT VARIABLES :
##
## MODEL_LOCATION                the root directory containing all the model checkpoint files (default: /gpfsssd/scratch/rech/ank/ucv88ce/projects/MultilingualCPC/checkpoints/inftrain)
## EVAL_LOCATION                 the root directory which will contain the eval files (default : /gpfsssd/scratch/rech/ank/ucv88ce/projects/MultilingualCPC/checkpoints/inftrain/)
## FAMILIES_LOCATION             the root directory containing source dataset with families (default:/gpfsssd/scratch/rech/cfs/commun/families)
## ZEROSPEECH_DATASET            the location of the zerospeech dataset used for evaluation (default: /gpfsssd/scratch/rech/ank/ucv88ce/data/zerospeech/zerospeech2017_dataset/test/)
## BASELINE_SCRIPTS              the location of the baseline script to use for feature extraction (default: /gpfsscratch/rech/cfs/uow84uh/InfTrain/zerospeech2021_baseline)
## FILE_EXTENSION                the extension to use as input in the feature extraction (default: wav)
## EVAL_NB_JOBS                  the number of jobs to use for evaluation (default: 20)
## GRU_LEVEL
## ABX_PY                        the script to use for abx evaluation (default: /gpfsdswork/projects/rech/ank/ucv88ce/repos/CPC_torch/cpc/eval/eval_ABX.py)
## BEST_EPOCH_PY                 the script to use to find the best epoch checkpoint (default: /gpfsdswork/projects/rech/ank/ucv88ce/projects/MultilingualCPC/utils/best_val_epoch.py)
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

MODEL_LOCATION="${MODEL_LOCATION:-/gpfsssd/scratch/rech/ank/ucv88ce/projects/MultilingualCPC/checkpoints/inftrain/}"
FAMILIES_LOCATION="${FAMILIES_LOCATION:-/gpfsssd/scratch/rech/cfs/commun/families/}"
EVAL_LOCATION="${EVAL_LOCATION:-/gpfsssd/scratch/rech/ank/ucv88ce/projects/MultilingualCPC/inftrain/}"
ZEROSPEECH_DATASET="${ZEROSPEECH_DATASET:-/gpfsssd/scratch/rech/ank/ucv88ce/data/zerospeech/zerospeech2017_dataset/test/}"
ABX_PY="${ABX_PY:-/gpfsdswork/projects/rech/ank/ucv88ce/repos/CPC_torch/cpc/eval/eval_ABX.py}"
BEST_EPOCH_PY="${BEST_EPOCH_PY:-/gpfsdswork/projects/rech/ank/ucv88ce/projects/MultilingualCPC/utils/best_val_epoch.py}"
FILE_EXT="${FILE_EXTENSION:-.wav}"
GRU_LEVEL="${GRU_LEVEL:-2}"
NB_JOBS="${EVAL_NB_JOBS:-20}"

PATH_TO_FAMILY=$1
shift

# check scripts locations

[ ! -f "$BEST_EPOCH_PY" ] && die "utils/best_val_epoch.py script was not found here : $BEST_EPOCH_PY"
[ ! -f "${ABX_PY}" ] && die "ABX script was not found at : ${ABX_PY}"

FAMILY_ID="${PATH_TO_FAMILY:${#FAMILIES_LOCATION}}"
CHECKPOINT_LOCATION="${MODEL_LOCATION}${FAMILY_ID}"
CPC_CHECKPOINT_FILE=""
#OUTPUT_LOCATION="${EVAL_LOCATION}${FAMILY_ID}"
OUTPUT_LOCATION="${CHECKPOINT_LOCATION}"

# Find best epoch checkpoint to use for evaluation
if [ -d "${CHECKPOINT_LOCATION}/cpc_small" ]; then
  BEST_EPOCH="$(python "$BEST_EPOCH_PY" --output-id --model_path "${CHECKPOINT_LOCATION}/cpc_small")"
  CPC_CHECKPOINT_FILE="${MODEL_LOCATION}${FAMILY_ID}/cpc_small/checkpoint_${BEST_EPOCH}.pt"
  OUTPUT_LOCATION="${OUTPUT_LOCATION}/cpc_small/ABX/${BEST_EPOCH}"
elif [ -d "${CHECKPOINT_LOCATION}/cpc_big" ]; then
  BEST_EPOCH="$(python "$BEST_EPOCH_PY" --output-id --model_path "${CHECKPOINT_LOCATION}/cpc_big")"
  CPC_CHECKPOINT_FILE="${MODEL_LOCATION}${FAMILY_ID}/cpc_big/checkpoint_${BEST_EPOCH}.pt"
  OUTPUT_LOCATION="${OUTPUT_LOCATION}/cpc_big/ABX/${BEST_EPOCH}"
else
  die "No CPC checkpoints found for family ${FAMILY_ID}"
fi

# Verify INPUTS
[ ! -d $ZEROSPEECH_DATASET ] && die "ZEROSPEECH_DATASET not found: $ZEROSPEECH_DATASET"
[ ! -d $MODEL_LOCATION ] && die "Model location does not exist: $MODEL_LOCATION"
[ ! -d $FAMILIES_LOCATION ] && die "Families location does not exist: $FAMILIES_LOCATION"
[ ! -d $PATH_TO_FAMILY ] && die "Given family was not found: $PATH_TO_FAMILY"

# we test only on 1s duration files
seconds="1s"

#--debug print values && exit
if [[ $DRY_RUN == "true" ]]; then
  echo "family-id: $FAMILY_ID"
  echo "zerospeech-dataset: $ZEROSPEECH_DATASET"
  echo "model-location: $MODEL_LOCATION"
  echo "families-location: $FAMILIES_LOCATION"
  echo "scripts: (abx;$ABX_PY), (epoch;$BEST_EPOCH_PY)"
  echo "checkpoint-file: $CPC_CHECKPOINT_FILE"
  echo "output-location: $OUTPUT_LOCATION"
  echo "file-extension: $FILE_EXT"
  echo "gru_level: $GRU_LEVEL"
  echo "nb-jobs: $NB_JOBS"
  echo "python $(which python)"
  echo "for langs in (french, english) using 1s files"
  for lang in french english; do
    PATH_ITEM_FILE="$ZEROSPEECH_DATASET/${lang}/${seconds}/${seconds}.item"
    PATH_OUT="$OUTPUT_LOCATION/${lang}"
    echo "==> python $ABX_PY from_checkpoint $CPC_CHECKPOINT_FILE $PATH_ITEM_FILE $ZEROSPEECH_DATASET --seq_norm --strict --file_extension $FILE_EXT --out $PATH_OUT"
  done
  exit 0
fi

module load sox
source activate inftrain

HOME=/gpfsdswork/projects/rech/ank/ucv88ce/
export PYTHONPATH=$HOME/repos/CPC_torch:$HOME/projects/MultilingualCPC/WavAugment:$PYTHONPATH

if [ ! -f ${CHECKPOINT_LOCATION}/cpc_small/done.state ] && [ ! -f ${CHECKPOINT_LOCATION}/cpc_big/done.state ]; then
    echo "=== The model hasn't finished training : no done.state in ${CHECKPOINT_LOCATION} . Not running the evaluation ==="
    exit 1
fi

for lang in french english; do
  PATH_ITEM_FILE="$ZEROSPEECH_DATASET/${lang}/${seconds}/${seconds}.item"
  LANG_DATASET="${ZEROSPEECH_DATASET}/${lang}/1s"
  PATH_OUT="$OUTPUT_LOCATION/${lang}"
  echo $PATH_OUT
  mkdir -p "$PATH_OUT"

  if [ ! -f ${PATH_OUT}/ABX_scores.json ]; then
      srun python $ABX_PY from_checkpoint $CPC_CHECKPOINT_FILE $PATH_ITEM_FILE --speaker-level 0 $LANG_DATASET --seq_norm --strict --file_extension $FILE_EXT --out $PATH_OUT
  fi
done
