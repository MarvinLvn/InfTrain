#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --partition=gpu_p2            # access to octo-gpus machines
#SBATCH --nodes=1                     # nombre de noeud
#SBATCH --gres=gpu:8                  # nombre de GPUs par nœud
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --exclusive
## Usage: ./evaluate_cpc.sh CPC_CHECKPOINT_FILE OUTPUT_LOCATION
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
##   CPC_CHECKPOINT_FILE         the path to the .pt file to use as the CPC model.
##   OUTPUT_LOCATION             the location to write result files.
##
## ENVIRONMENT VARIABLES
##
## ZEROSPEECH_DATASET            the location of the zerospeech dataset used for evaluation (default: /scratch1/projects/zerospeech/2021/zerospeech2021_dataset)
## BASELINE_SCRIPTS              the location of the baseline script to use for feature extraction (default: ../external_code/zerospeech2021_baseline)
## FILE_EXTENSION                the extension to use as input in the feature extraction (default: wav)
## EVAL_NB_JOBS                  the number of jobs to use for evaluation (default: 20)
## GRU_LEVEL
##
## More info:
## https://github.com/bootphon/zerospeech2021_baseline
## https://docs.google.com/spreadsheets/d/1pcT_6YLdQ5Oa2pO21mRKzzU79ZPUZ-BfU2kkXg2mayE/edit?usp=drive_web&ouid=112305914309228781110

echo "This script hasn't been tested."
exit 0

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
[ $# -lt 2 ] && usage

ZEROSPEECH_DATASET="${ZEROSPEECH_DATASET:-/scratch1/projects/zerospeech/2021/zerospeech2021_dataset}"
BASELINE_SCRIPTS="${BASELINE_SCRIPTS:-$(dirname "$here")/external_code/zerospeech2021_baseline}"
FILE_EXT="${FILE_EXTENSION:-wav}"
GRU_LEVEL="${GRU_LEVEL:-2}"
NB_JOBS="${EVAL_NB_JOBS:-20}"

CPC_CHECKPOINT_FILE=$1
OUTPUT_LOCATION=$2
shift; shift;

#--debug
#echo "zerospeech-dataset: $ZEROSPEECH_DATASET"
#echo "baseline-scripts: $BASELINE_SCRIPTS"
#echo "checkpoint-file: $CPC_CHECKPOINT_FILE"
#echo "output-location: $OUTPUT_LOCATION"
#echo "file-extension: $FILE_EXT"

# Verify INPUTS
[ ! -d $ZEROSPEECH_DATASET ] && die "ZEROSPEECH_DATASET not found: $ZEROSPEECH_DATASET"
[ ! -d $BASELINE_SCRIPTS ] && die "BASELINE_SCRIPTS not found: $BASELINE_SCRIPTS"
[ ! -f $CPC_CHECKPOINT_FILE ] && [ "${CPC_CHECKPOINT_FILE: -3}" == ".pt" ] && die "Checkpoint file given does not exist or is not a valid .pt file: $CPC_CHECKPOINT_FILE"

mkdir -p $OUTPUT_LOCATION/features/phonetic/{'dev-clean','dev-other','test-clean','test-other'}

# check that script exists
[ ! -f "${BASELINE_SCRIPTS}/scripts/build_CPC_features.py" ] && die "CPC feature build was not found in ${BASELINE_SCRIPTS}/scripts"

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