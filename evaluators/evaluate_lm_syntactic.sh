#!/bin/bash
#SBATCH --account=xdz@gpu
#SBATCH --partition=prepost           # access to octo-gpus machines
#SBATCH --nodes=1                     # nombre de noeud
##SBATCH --gres=gpu:1                  # nombre de GPUs par nœud
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --exclusive
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
## BASELINE_SCRIPTS              the location of the baseline script to use for feature extraction (default: ../external_code/zerospeech2021_baseline)
## FILE_EXTENSION                the extension to use as input in the feature extraction (default: wav)
## EVAL_NB_JOBS                  the number of jobs to use for evaluation (default: 20)
## KIND                          the partition of the zerospeech dataset on which the evaluation is done (default: dev test)
## OUTPUT_LOCATION               the location to write result files (default: family_id/cpc_{small, big})
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
BASELINE_SCRIPTS="${BASELINE_SCRIPTS:-../external_code/zerospeech2021_baseline}"
FILE_EXT="${FILE_EXTENSION:-wav}"
NB_JOBS="${EVAL_NB_JOBS:-20}"
KIND=('dev')

FAMILY_ID=$(echo "$(cd "$(dirname "$1")"; pwd)/$(basename "$1")")

# TODO : cpc_small or cpc_big

CLUSTERING_CHECKPOINT_FILE="$FAMILY_ID/cpc_small/clustering_kmeans50/clustering_CPC_big_kmeans50.pt"
OUTPUT_LOCATION="$FAMILY_ID/cpc_small"

# OUTPUT_LOCATION=$2
shift; shift;



# Verify INPUTS
[ ! -d $ZEROSPEECH_DATASET ] && die "ZEROSPEECH_DATASET not found: $ZEROSPEECH_DATASET"
[ ! -d $BASELINE_SCRIPTS ] && die "BASELINE_SCRIPTS not found: $BASELINE_SCRIPTS"
[ ! -f $CLUSTERING_CHECKPOINT_FILE ] && [ "${CLUSTERING_CHECKPOINT_FILE: -3}" == ".pt" ] && die "Checkpoint file given does not exist or is not a valid .pt file: $CPC_CHECKPOINT_FILE"


# check that script exists
[ ! -f "${BASELINE_SCRIPTS}/scripts/quantize_audio.py" ] && die "Quantize audio was not found in ${BASELINE_SCRIPTS}/scripts"
[ ! -f "${BASELINE_SCRIPTS}/scripts/build_1hot_features.py" ] && die "Build 1-hot features was not found in ${BASELINE_SCRIPTS}/scripts"
[ ! -f "${BASELINE_SCRIPTS}/scripts/compute_proba_BERT.py" ] && die "Compute proba BERT was not found in ${BASELINE_SCRIPTS}/scripts"
[ ! -f "${BASELINE_SCRIPTS}/scripts/compute_proba_LSTM.py" ] && die "Compute proba LSTM was not found in ${BASELINE_SCRIPTS}/scripts"

# -- Extract quantized units on zerospeech20201/syntactic

mkdir -p $OUTPUT_LOCATION/features/syntactic/{'dev','test'}

for item in ${KIND[*]}
do
  datafiles="${ZEROSPEECH_DATASET}/syntactic/${item}"
  output="${OUTPUT_LOCATION}/features/syntactic/${item}"
  python "${BASELINE_SCRIPTS}/scripts/quantize_audio.py" "${CLUSTERING_CHECKPOINT_FILE}" "${datafiles}" "${output}" --file_extension $FILE_EXT
done


# -- Compute pseudo-probabilities (bert or lstm) depending on the model

mkdir -p $OUTPUT_LOCATION/features/probabailities/{'dev','test'}


# python compute_proba_{BERT,LSTM}.py pathQuantizedUnits pathOutputFile pathBERTCheckpoint
# bert : > 400h
# lstm : < 400h
# 400h : both models TODO

duration=$(echo ${CLUSTERING_CHECKPOINT_FILE} | cut -d'/' -f8 | sed "s/\h//g")

if [ $duration -lt 401 ]
then
  MODEL="LSTM"
else
  MODEL="BERT"
fi


for item in ${KIND[*]}
do
  quantized="$OUTPUT_LOCATION/features/syntactic/${item}/quantized_outputs.txt"
  output="$OUTPUT_LOCATION/features/syntactic/$item.txt"
  bert_checkpoint="$OUTPUT_LOCATION/LSTM/LSTM_CPC_big_kmeans50.pt" # checkpoint of the model in part 3 of trainig
  python "${BASELINE_SCRIPTS}/scripts/compute_proba_${MODEL}.py" "${quantized}" "${output}" "${bert_checkpoint}" --cpu
done


# Compute SWUGGY
# -- Prepare for evaluation

FEATURES_LOCATION="${OUTPUT_LOCATION}/features"
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
# export ZEROSPEECH2021_TEST_GOLD=$ZEROSPEECH_DATASET

echo "evaluation with zerospeech2021-evaluate"

zerospeech2021-evaluate --no-phonetic --no-lexical --no-semantic --njobs $NB_JOBS -o "$OUTPUT_LOCATION/scores/swuggy" $ZEROSPEECH_DATASET $FEATURES_LOCATION


# cleanup
rm -r $OUTPUT_LOCATION/features
