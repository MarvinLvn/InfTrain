#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --nodes=1                     # nombre de noeud
#SBATCH --gres=gpu:1                  # nombre de GPUs par nœud
#SBATCH --time=15:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
## Usage: ./evaluate_lm_lexical.sh PATH/TO/FAMILY_ID bert en testset_1
##
## 1) Extract quantized units (scripts/quantize_audio.py) on $ALL_CCFRSCRATCH/swuggy_hadrien/lexical
## 2) Compute pseudo-probabilities scripts/compute_proba_BERT.py or scripts/compute_proba_LSTM.py depending on the model
## 3) Compute sWUGGY (lexical score)
##
## Example:
##
## ./evaluate_lm_lexical.sh path/to/family_id model_type
##
## Parameters:
##
##   PATH/TO/FAMILY              the path to the family id.
##   MODEL_TYPE                  language model to evaluate (bert|bert_small|lstm)
##   TEST_LANGUAGE               test language to be considered: en or fr
##   TEST_SHARE                  subfolder to consider when evaluating: testset_{1,2,3,4,8,16,32,64}
##
## ENVIRONMENT VARIABLES
##
## MODEL_LOCATION                the root directory containing all the model checkpoint files (default: /gpfsscratch/rech/cfs/commun/InfTrain_models)
## FAMILIES_LOCATION             the root directory containing source dataset with families (default: /gpfsscratch/rech/cfs/commun/families)
## TEST_PATH                     the location of the test set used for evaluation (default: /gpfsscratch/rech/cfs/commun/swuggy_hadrien)
## BASELINE_SCRIPTS              the location of the baseline script to use for feature extraction (default: ../utils)
## FILE_EXTENSION                the extension to use as input in the feature extraction (default: wav)
## EVAL_NB_JOBS                  the number of jobs to use for evaluation (default: 20)
## KIND                          the partition of the zerospeech dataset on which the evaluation is done (default: dev test)
## OUTPUT_LOCATION               the location to write features files (default: $JOBSCRATCH on Jean-Zay)
##
## More info:
## https://github.com/bootphon/zerospeech2021_baseline
## https://docs.google.com/spreadsheets/d/1pcT_6YLdQ5Oa2pO21mRKzzU79ZPUZ-BfU2kkXg2mayE/edit?usp=drive_web&ouid=112305914309228781110

# check only parameters without running eval
DRY_RUN="${DRY_RUN:-false}"

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
[ $# -lt 3 ] && usage

# paths
MODEL_LOCATION="${MODEL_LOCATION:-/gpfsscratch/rech/cfs/commun/InfTrain_models/big}"
FAMILIES_LOCATION="${FAMILIES_LOCATION:-/gpfsscratch/rech/cfs/commun/families}"

BASELINE_SCRIPTS="${BASELINE_SCRIPTS:-utils}"
FILE_EXT="${FILE_EXTENSION:-wav}"
NB_JOBS="${EVAL_NB_JOBS:-20}"
KIND=('dev')

PATH_TO_FAMILY=$1
MODEL=$2
TEST_LANGUAGE=$3
TEST_SHARE=$4

TEST_PATH="${TEST_PATH:-/gpfsscratch/rech/cfs/commun/swuggy_hadrien/lexical/${TEST_LANGUAGE}/${TEST_SHARE}}"
FAMILY_ID="${PATH_TO_FAMILY#${FAMILIES_LOCATION}}"
CHECKPOINT_LOCATION="${MODEL_LOCATION}${FAMILY_ID}"

output_file=$CHECKPOINT_LOCATION/$MODEL/scores/swuggy/${TEST_LANGUAGE}/swuggy_${TEST_LANGUAGE}_${TEST_SHARE}/score_lexical_dev_by_pair.csv
if [ -f $output_file ]; then
  echo "$output_file found. Not running anything."
  exit
fi;


if [ -d "${CHECKPOINT_LOCATION}/cpc_small" ]; then
  CPC="cpc_small"
elif [ -d "${CHECKPOINT_LOCATION}/cpc_big" ]; then
  CPC="cpc_big"
else
  die "No CPC checkpoints found for family ${CHECKPOINT_LOCATION}"
fi

OUTPUT_LOCATION="$JOBSCRATCH$CPC"

if [ -d "${CHECKPOINT_LOCATION}/kmeans50" ]; then
  CLUSTERING_CHECKPOINT_FILE="$CHECKPOINT_LOCATION/kmeans50/checkpoint_last.pt"
else
  die "No CPC-kmeans checkpoints found for family ${CHECKPOINT_LOCATION}"
fi

if [ -d "${CHECKPOINT_LOCATION}/$MODEL" ]; then
  LM_CHECKPOINT_FILE="$CHECKPOINT_LOCATION/$MODEL/checkpoint_best.pt"
else
  die "No ${MODEL} checkpoints found for family ${CHECKPOINT_LOCATION}"
fi


# Verify INPUTS
[ ! -d $TEST_PATH ] && die "TEST_PATH not found: $TEST_PATH"
[ ! -d $BASELINE_SCRIPTS ] && die "BASELINE_SCRIPTS not found: $BASELINE_SCRIPTS"
[ ! -f $CLUSTERING_CHECKPOINT_FILE ] && [ "${CLUSTERING_CHECKPOINT_FILE: -3}" == ".pt" ] && die "Checkpoint file given does not exist or is not a valid .pt file: $CPC_CHECKPOINT_FILE"


# check that script exists
[ ! -f "${BASELINE_SCRIPTS}/quantize_audio.py" ] && die "Quantize audio was not found in ${BASELINE_SCRIPTS}"
[ ! -f "${BASELINE_SCRIPTS}/compute_proba_BERT.py" ] && die "Compute proba BERT was not found in ${BASELINE_SCRIPTS}"
[ ! -f "${BASELINE_SCRIPTS}/compute_proba_LSTM.py" ] && die "Compute proba LSTM was not found in ${BASELINE_SCRIPTS}"

#--debug print values && exit
if [[ $DRY_RUN == "true" ]]; then
  echo "-------------- VARIABLES ---------------------------"
  echo "family-id: $FAMILY_ID"
  echo "test-set: $TEST_PATH"
  echo "model-location: $MODEL_LOCATION"
  echo "families-location: $FAMILIES_LOCATION"
  echo "checkpoint-location: $CHECKPOINT_LOCATION"
  echo "clusturing-checkpoint-file: $CLUSTERING_CHECKPOINT_FILE"
  echo "lm-checkpoint-file: $LM_CHECKPOINT_FILE"
  echo "output-location: $OUTPUT_LOCATION"
  echo "file-extension: $FILE_EXT"
  echo "python $(which python)"
  exit 0
fi

# -- Extract quantized units on test set
mkdir -p $OUTPUT_LOCATION/features_lex/${TEST_LANGUAGE}/${TEST_SHARE}/lexical/{'dev','test'}

for item in ${KIND[*]}
do
  datafiles="${TEST_PATH}/lexical/$item"
  output="${OUTPUT_LOCATION}/features_lex/${TEST_LANGUAGE}/${TEST_SHARE}/lexical/$item"
  python "${BASELINE_SCRIPTS}/quantize_audio.py" "${CLUSTERING_CHECKPOINT_FILE}" "${datafiles}" "${output}" --file_extension $FILE_EXT
done


# -- Compute pseudo-probabilities (bert or lstm) depending on the model
MODEL_TYPE=${MODEL/_small/}
MODEL_TYPE=${MODEL_TYPE/_sbm_none/}
MODEL_TYPE=${MODEL_TYPE/_sbm_complete/}
MODEL_TYPE=${MODEL_TYPE/_sbm_eos/}
MODEL_TYPE=${MODEL_TYPE^^}
for item in ${KIND[*]}
do
  quantized="$OUTPUT_LOCATION/features_lex/${TEST_LANGUAGE}/${TEST_SHARE}/lexical/$item/quantized_outputs.txt"
  output="$OUTPUT_LOCATION/features_lex/${TEST_LANGUAGE}/${TEST_SHARE}/lexical/$item.txt"
  python "${BASELINE_SCRIPTS}/compute_proba_${MODEL_TYPE}.py" "${quantized}" "${output}" "${LM_CHECKPOINT_FILE}"
done


# Compute SWUGGY
# -- Prepare for evaluation

FEATURES_LOCATION="${OUTPUT_LOCATION}/features_lex/${TEST_LANGUAGE}/${TEST_SHARE}"
# meta.yaml (required by zerospeech2021-evaluate)
cat <<EOF > $FEATURES_LOCATION/meta.yaml
author: infantSim Train Eval
affiliation: EHESS, ENS, PSL Research Univerity, CNRS and Inria
description: >
  CPC-big (trained on librispeech 960), kmeans (trained on librispeech 100),
  BERT (trained on librispeech 960 encoded with the quantized units). See
  https://zerospeech.com/2021 for more details.
open_source: true
train_set: inftrain families
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
  export ZEROSPEECH2021_TEST_GOLD=$TEST_PATH
fi;

zerospeech2021-evaluate --no-phonetic --no-syntactic --no-semantic --njobs $NB_JOBS -o "$OUTPUT_LOCATION/scores/swuggy_${TEST_LANGUAGE}_${TEST_SHARE}" $TEST_PATH $FEATURES_LOCATION

# copy the score on $SCRATCH
mkdir -p $CHECKPOINT_LOCATION/$MODEL/scores/swuggy/${TEST_LANGUAGE}
cp -r $OUTPUT_LOCATION/scores/swuggy_${TEST_LANGUAGE}_${TEST_SHARE} $CHECKPOINT_LOCATION/$MODEL/scores/swuggy/${TEST_LANGUAGE}
