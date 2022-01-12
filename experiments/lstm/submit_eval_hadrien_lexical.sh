#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --output=../logs/eval_hadrien_lexical_%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --array=253-254%254
#SBATCH --hint=nomultithread          # hyperthreading desactive

# This script works a bit differently than others
# It must be called as:
# sbatch submit_eval_hadrien_lexical.sh <MODEL_TYPE> <TEST_LANG> <SHARE>
# where:
#   <MODEL_TYPE>        the type of model that needs to be evaluate: (bert|bert_small|lstm)
#   <TEST_LANG>         the test language: (en|fr)
#   <SHARE>             the frequency band to be considered: (1|4|8|16|32|64) with 64 corresponding to words present in the 64 training sets of 50 hours, etc.

source activate inftrain
module load sox

if [[ $# -ne 3 ]]; then
  echo "sbatch submit_eval_hadrien_lexical.sh lstm en testset_1"
fi;

ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /gpfsscratch/rech/cfs/uow84uh/InfTrain/experiments/experiments_txt/bert_experiments.txt)
cd ../..

./evaluators/evaluate_lm_hadrien_lexical.sh ${ARGS} $1 $2 $3
