#!/bin/bash
#SBATCH --account=cfs@gpu
##SBATCH --partition=prepost           # access to octo-gpus machines
#SBATCH --nodes=1                     # nombre de noeud
#SBATCH --gres=gpu:1                  # nombre de GPUs par nœud
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
## Usage: ./evaluate_lm_syntactic.sh PATH/TO/FAMILY_ID
##
## 1) Extract quantized units (scripts/quantize_audio.py) on zerospeech2021/syntactic
## 2) Compute pseudo-probabilities scripts/compute_proba_BERT.py or scripts/compute_proba_LSTM.py depending on the model
## 3) Compute sBLIMP (syntactic score)
##
## Example:
##
## ./evaluate_lm_syntactic.sh path/to/family_id
##
## Parameters:
##
##   PATH/TO/FAMILY              the path to the family id.
##
## ENVIRONMENT VARIABLES
##
## ZEROSPEECH_DATASET            the location of the zerospeech dataset used for evaluation (default: /gpfsscratch/rech/cfs/commun/zerospeech2021_dataset)
## BASELINE_SCRIPTS              the location of the baseline script to use for feature extraction (default: ../utils)
## FILE_EXTENSION                the extension to use as input in the feature extraction (default: wav)
## EVAL_NB_JOBS                  the number of jobs to use for evaluation (default: 20)
## KIND                          the partition of the zerospeech dataset on which the evaluation is done (default: dev test)
## OUTPUT_LOCATION               the location to write features files (default: $JOBSCRATCH on Jean-Zay)
##
## More info:
## https://github.com/bootphon/zerospeech2021_baseline
## https://docs.google.com/spreadsheets/d/1pcT_6YLdQ5Oa2pO21mRKzzU79ZPUZ-BfU2kkXg2mayE/edit?usp=drive_web&ouid=112305914309228781110



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

ZEROSPEECH_DATASET="${ZEROSPEECH_DATASET:-/gpfsscratch/rech/cfs/commun/zerospeech2021_dataset}"
BASELINE_SCRIPTS="${BASELINE_SCRIPTS:-utils}"
FILE_EXT="${FILE_EXTENSION:-wav}"
NB_JOBS="${EVAL_NB_JOBS:-20}"
KIND=('dev')

FAMILY_ID=$(echo "$(cd "$(dirname "$1")"; pwd)/$(basename "$1")")


if [ -d "${FAMILY_ID}/cpc_small" ]; then
  CPC="cpc_small"
  MODEL="LSTM"
elif [ -d "${FAMILY_ID}/cpc_big" ]; then
  CPC="cpc_big"
  MODEL="BERT"
else
  die "No CPC checkpoints found for family ${FAMILY_ID}"
fi

OUTPUT_LOCATION="$JOBSCRATCH/$CPC"

if [ -d "${FAMILY_ID}/kmeans50" ]; then
  CLUSTERING_CHECKPOINT_FILE="$FAMILY_ID/kmeans50/checkpoint_last.pt"
else
  die "No CPC-kmeans checkpoints found for family ${FAMILY_ID}"
fi

if [ -d "${FAMILY_ID}/${MODEL,,}" ]; then
  LM_CHECKPOINT_FILE="$FAMILY_ID/${MODEL,,}/checkpoint_best.pt"
else
  die "No ${MODEL} checkpoints found for family ${FAMILY_ID}"
fi


# Verify INPUTS
[ ! -d $ZEROSPEECH_DATASET ] && die "ZEROSPEECH_DATASET not found: $ZEROSPEECH_DATASET"
[ ! -d $BASELINE_SCRIPTS ] && die "BASELINE_SCRIPTS not found: $BASELINE_SCRIPTS"
[ ! -f $CLUSTERING_CHECKPOINT_FILE ] && [ "${CLUSTERING_CHECKPOINT_FILE: -3}" == ".pt" ] && die "Checkpoint file given does not exist or is not a valid .pt file: $CPC_CHECKPOINT_FILE"


# check that script exists
[ ! -f "${BASELINE_SCRIPTS}/quantize_audio.py" ] && die "Quantize audio was not found in ${BASELINE_SCRIPTS}"
[ ! -f "${BASELINE_SCRIPTS}/compute_proba_BERT.py" ] && die "Compute proba BERT was not found in ${BASELINE_SCRIPTS}"
[ ! -f "${BASELINE_SCRIPTS}/compute_proba_LSTM.py" ] && die "Compute proba LSTM was not found in ${BASELINE_SCRIPTS}"

# -- Extract quantized units on zerospeech20201/syntactic

mkdir -p $OUTPUT_LOCATION/features_syn/syntactic/{'dev','test'}

for item in ${KIND[*]}
do
  datafiles="${ZEROSPEECH_DATASET}/syntactic/${item}"
  output="${OUTPUT_LOCATION}/features_syn/syntactic/${item}"
  python "${BASELINE_SCRIPTS}/quantize_audio.py" "${CLUSTERING_CHECKPOINT_FILE}" "${datafiles}" "${output}" --file_extension $FILE_EXT
done


# -- Compute pseudo-probabilities (bert or lstm) depending on the model

ARGUMENTS=""
if [ "$MODEL" == "LSTM" ] ; then
  ARGUMENTS="--batchSize=64"
else
  ARGUMENTS="None"
fi;

for item in ${KIND[*]}
do
  quantized="$OUTPUT_LOCATION/features_syn/syntactic/${item}/quantized_outputs.txt"
  output="$OUTPUT_LOCATION/features_syn/syntactic/$item.txt"
  if [ "$ARGUMENTS" == "None" ] ; then
    python "${BASELINE_SCRIPTS}/compute_proba_${MODEL}.py" "${quantized}" "${output}" "${LM_CHECKPOINT_FILE}"
  else
    python "${BASELINE_SCRIPTS}/compute_proba_${MODEL}.py" "${quantized}" "${output}" "${LM_CHECKPOINT_FILE}" "${ARGUMENTS}"
  fi
done


# Compute SBLIMP
# -- Prepare for evaluation

FEATURES_LOCATION="${OUTPUT_LOCATION}/features_syn"
# meta.yaml (required by zerospeech2021-evaluate)
cat <<EOF > $FEATURES_LOCATION/meta.yaml
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
if [[ ${KIND[*]} =~ (^|[[:space:]])"test"($|[[:space:]]) ]] ; then
  export ZEROSPEECH2021_TEST_GOLD=$ZEROSPEECH_DATASET
fi;

zerospeech2021-evaluate --no-phonetic --no-lexical --no-semantic --njobs $NB_JOBS -o "$OUTPUT_LOCATION/scores/sblimp" $ZEROSPEECH_DATASET $FEATURES_LOCATION

# copy the score on $SCRATCH
mkdir -p $FAMILY_ID/${MODEL,,}/scores/sblimp
cp -r $OUTPUT_LOCATION/scores/sblimp $FAMILY_ID/${MODEL,,}/scores
