#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --time=10:00:00
## Usage: ./evaluate_cpc.sh /path/to/duration/family_id
##
## 1) Extract CPC representations on zerospeech2021/phonetic (script : scripts/build_CPC_features.py)
## 2) Compute ABX error rate (command : zerospeech2021-evaluate)
##
## Example:
##
## ./evaluate_cpc.sh path/to/checkpoints/cpc/file.pt path/to/output
##
## Parameters:
##
##   FAMILY_PATH (ID)            the path to the family that we want to evaluate.
##
## ENVIRONMENT VARIABLES :
##
## OUTPUT_LOCATION               the location to write result files (default: /gpfsscratch/rech/cfs/commun/InfTrain_models_eval).
## MODEL_LOCATION                the root directory containing all the model checkpoint files (default: /gpfsscratch/rech/cfs/commun/InfTrain_models)
## FAMILIES_LOCATION             the root directory containing source dataset with families (default: /gpfsscratch/rech/cfs/commun/families)
## ZEROSPEECH_DATASET            the location of the zerospeech dataset used for evaluation (default: /gpfsscratch/rech/cfs/commun/zerospeech2021_dataset)
## BASELINE_SCRIPTS              the location of the baseline script to use for feature extraction (default: /gpfsscratch/rech/cfs/uow84uh/InfTrain/zerospeech2021_baseline)
## FILE_EXTENSION                the extension to use as input in the feature extraction (default: wav)
## EVAL_NB_JOBS                  the number of jobs to use for evaluation (default: 20)
## GRU_LEVEL
##
## More info:
## https://github.com/bootphon/zerospeech2021_baseline
## https://docs.google.com/spreadsheets/d/1pcT_6YLdQ5Oa2pO21mRKzzU79ZPUZ-BfU2kkXg2mayE/edit?usp=drive_web&ouid=112305914309228781110

#echo "This script hasn't been tested."
#exit 0

# todo check slurm parameters & options
# --- Various utility functions & variables

# absolute path to the directory where this script is
here="$(cd $(dirname "${BASH_SOURCE[0]}") > /dev/null 2>&1 && pwd)"

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

OUTPUT_LOCATION="${OUTPUT_LOCATION:-/gpfsscratch/rech/cfs/commun/InfTrain_models_eval}"
MODEL_LOCATION="${MODEL_LOCATION:-/gpfsscratch/rech/cfs/commun/InfTrain_models}"
FAMILIES_LOCATION="${FAMILIES_LOCATION:-/gpfsscratch/rech/cfs/commun/families}"
ZEROSPEECH_DATASET="${ZEROSPEECH_DATASET:-/gpfsscratch/rech/cfs/commun/zerospeech2021_dataset}"
BASELINE_SCRIPTS="${BASELINE_SCRIPTS:-/gpfsscratch/rech/cfs/uow84uh/InfTrain/zerospeech2021_baseline}"
FILE_EXT="${FILE_EXTENSION:-wav}"
GRU_LEVEL="${GRU_LEVEL:-2}"
NB_JOBS="${EVAL_NB_JOBS:-20}"

PATH_TO_FAMILY=$1
shift;

# check scripts locations
BEST_EPOCH_PY="$(dirname $here)/utils/best_val_epoch.py"

[ ! -f "$BEST_EPOCH_PY" ] && die "utils/best_val_epoch.py script was not found here : $BEST_EPOCH_PY"
[ ! -f "${BASELINE_SCRIPTS}/scripts/build_CPC_features.py" ] && die "CPC feature build was not found in ${BASELINE_SCRIPTS}/scripts"
[ ! -x "$(command -v zerospeech2021-evaluate)" ] && die "zerospeech2021-evaluate command was not found, it needs to be installed to allow evaluation"


FAMILY_ID="${PATH_TO_FAMILY#${FAMILIES_LOCATION}}"
CHECKPOINT_LOCATION="${MODEL_LOCATION}${FAMILY_ID}"
CPC_CHECKPOINT_FILE=""

# Find best epoch checkpoint to use for evaluation
if [ -d "${CHECKPOINT_LOCATION}/cpc_small" ]; then
  BEST_EPOCH="$(python "$BEST_EPOCH_PY" --output-id --model_path "${CHECKPOINT_LOCATION}/cpc_small")"
  CPC_CHECKPOINT_FILE="${MODEL_LOCATION}${FAMILY_ID}/cpc_small/checkpoint_${BEST_EPOCH}.pt"
elif [ -d "${CHECKPOINT_LOCATION}/cpc_big" ]; then
  BEST_EPOCH="$(python "$BEST_EPOCH_PY" --output-id --model_path "${CHECKPOINT_LOCATION}/cpc_big")"
  CPC_CHECKPOINT_FILE="${MODEL_LOCATION}${FAMILY_ID}/cpc_big/checkpoint_${BEST_EPOCH}.pt"
else
  die "No CPC checkpoints found for family ${FAMILY_ID}"
fi

#--debug
echo "family-id: $FAMILY_ID"
echo "zerospeech-dataset: $ZEROSPEECH_DATASET"
echo "model-location: $MODEL_LOCATION"
echo "families-location: $FAMILIES_LOCATION"
echo "baseline-scripts: $BASELINE_SCRIPTS"
echo "checkpoint-file: $CPC_CHECKPOINT_FILE"
echo "output-location: $OUTPUT_LOCATION"
echo "file-extension: $FILE_EXT"
echo "gru_level: $GRU_LEVEL"
echo "nb-jobs: $NB_JOBS"

# Verify INPUTS
[ ! -d $ZEROSPEECH_DATASET ] && die "ZEROSPEECH_DATASET not found: $ZEROSPEECH_DATASET"
[ ! -d $BASELINE_SCRIPTS ] && die "BASELINE_SCRIPTS path not found: $BASELINE_SCRIPTS"
[ ! -d $MODEL_LOCATION ] && die "Model location does not exist: $MODEL_LOCATION"
[ ! -d $FAMILIES_LOCATION ] && die "Families location does not exist: $FAMILIES_LOCATION"
[ ! -d $PATH_TO_FAMILY ] && die "Given family was not found: $PATH_TO_FAMILY"

OUTPUT_LOCATION="${OUTPUT_LOCATION}${FAMILY_ID}"

mkdir -p $OUTPUT_LOCATION/features/phonetic/{'dev-clean','dev-other','test-clean','test-other'}

exit 0
for item in 'dev-clean' 'dev-other' 'test-clean' 'test-other'
do
  datafiles="${ZEROSPEECH_DATASET}/phonetic/${item}"
  output="${OUTPUT_LOCATION}/features/phonetic/${item}"
  python "${BASELINE_SCRIPTS}/scripts/build_CPC_features.py" "${CPC_CHECKPOINT_FILE}" "${datafiles}" "${output}" --file-extension $FILE_EXT --gru_level $GRU_LEVEL
done


# -- Prepare for evaluation

FEATURES_LOCATION="${OUTPUT_LOCATION}/features"
# meta.yaml (required by zerospeech2021-evaluate)
cat <<EOF > $FEATURES_LOCATION
author: infantSim Train Eval
affiliation: EHESS, ENS, PSL Research Univerity, CNRS and Inria
description: >
  CPC-big (trained on librispeech 960), kmeans (trained on librispeech 100),
  BERT (trained on librispeech 960 encoded with the quantized units). See
  https://zerospeech.com/2021 for more details.
open_source: true
train_set: librispeech 100 and 960
gpu_budget: 1536
parameters:
  phonetic:
    metric: cosine
    frame_shift: 0.01
  semantic:
    metric: cosine
    pooling: max
EOF

# required by zerospeech2021-evaluate to allow test evaluation
export ZEROSPEECH2021_TEST_GOLD=$ZEROSPEECH_DATASET

zerospeech2021-evaluate --no-lexical --no-syntactic --no-semantic --njobs $NB_JOBS -o "$OUTPUT_LOCATION/scores/phonetic" $ZEROSPEECH_DATASET $FEATURES_LOCATION

# clean feature files
[ -d "${FEATURES_LOCATION}" ] && rm -rf "${FEATURES_LOCATION}"
