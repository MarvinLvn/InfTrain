#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --partition=gpu_p2
#SBATCH --nodes=4                     # nombre de noeud
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8                  # nombre de GPUs par nœud
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters. Please respect :"
  echo "./train_bert.sh /path/to/database/containing/wav/files"
  echo "Example :"
  echo "./train_bert.sh /gpfsscratch/rech/cfs/commun/families/EN/50h/00"
  exit
fi

# to be deleted
MODELS_PATH=${ALL_CCFRSCRATCH}/InfTrain_models
BASELINE_SCRIPTS=utils
FAIRSEQ_SCRIPTS=fairseq
FILE_EXT=.wav

# Arguments
PATH_DB=$1
SHARE=$(basename $PATH_DB)
SIZE=$(basename $(dirname $PATH_DB))
LANGUAGE=$(basename $(dirname $(dirname $PATH_DB)))
PATH_KMEANS=${MODELS_PATH}/${LANGUAGE}/${SIZE}/${SHARE}/kmeans50/checkpoint_last.pt

if [ ! -f $PATH_KMEANS ]; then
  echo "Can't find k-means model : $PATH_PATH_KMEANS".
  echo "Please, train k-means before training language models"
fi;

# 1) Quantize audio + split train val test
TRAIN_SET=$PATH_DB
OUTPUT_LOCATION="$JOBSCRATCH/${LANGUAGE}_${SIZE}_${SHARE}"
OUTPUT=$OUTPUT_LOCATION/quantized_train
#python "${BASELINE_SCRIPTS}/quantize_audio.py" "${PATH_KMEANS}" "${TRAIN_SET}" "${OUTPUT}" --file_extension $FILE_EXT --resume

# check nb files
NB_FILES=$(find $TRAIN_SET -name "*$FILE_EXT" | wc -l)
NB_LINES=$(cat $OUTPUT_LOCATION/quantized_train/quantized_outputs.txt | wc -l)

#echo "Found $NB_FILES files".
#echo "$(($NB_LINES+1)) sequences have been quantized"
#
#if [ $(($NB_LINES+1)) -ne $NB_FILES ]; then
#  echo "Wrong number of quantized sequences. Stopped."
#  exit
#fi;

# + convert format
#cat $OUTPUT_LOCATION/quantized_train/quantized_outputs.txt | awk '{print $2}' | sed 's/,/ /g' > $OUTPUT_LOCATION/quantized_train/quantized_outputs_2.txt
# + split train/val/test
#python ${BASELINE_SCRIPTS}/split_train_val_test_lm.py --input_file $OUTPUT_LOCATION/quantized_train/quantized_outputs_2.txt

# 2) Fairseq preprocess
#fairseq-preprocess --only-source \
#      --trainpref $OUTPUT_LOCATION/quantized_train/fairseq_train.txt \
#      --validpref $OUTPUT_LOCATION/quantized_train/fairseq_val.txt \
#      --testpref $OUTPUT_LOCATION/quantized_train/fairseq_test.txt \
#      --destdir $OUTPUT_LOCATION/fairseq_bin_data \
#      --workers 20

# 3) Train models
MODEL_OUTPUT=${MODELS_PATH}/${LANGUAGE}/${SIZE}/${SHARE}/bert
mkdir -p $MODEL_OUTPUT
#cp $OUTPUT_LOCATION/fairseq_bin_data/dict.txt $MODEL_OUTPUT
#cp -r $OUTPUT_LOCATION/fairseq_bin_data $MODEL_OUTPUT

SPAN_SIZE=10
MAX_TOKENS=4096
GPU_PER_TASK=8
CPU_PER_TASK=10
TASKS_PER_NODE=1
NODES=4
TOTAL_GPU=$((GPU_PER_TASK * TASKS_PER_NODE * NODES))
DISTRIBUTED_PORT=52663
UPDATE_FREQ=$((128 / TOTAL_GPU))

start=`date +%s`
srun python /gpfsscratch/rech/cfs/uow84uh/InfTrain/fairseq/train.py --fp16 $MODEL_OUTPUT/fairseq_bin_data \
--save-dir ${MODEL_OUTPUT} \
--keep-last-epochs 1 \
--tensorboard-logdir tensorboard \
--train-subset train \
--num-workers 4 \
--task masked_lm --criterion masked_lm \
--arch roberta_base \
--sample-break-mode none --tokens-per-sample 3072 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
--lr-scheduler polynomial_decay --lr 0.0005 --total-num-update 250000 --warmup-updates 10000 \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--mask-multiple-length $SPAN_SIZE --mask-prob 0.5 --mask-stdev $SPAN_SIZE \
--max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ --max-update 250000 \
--seed 5 --log-format simple --log-interval 10 --skip-invalid-size-inputs-valid-test \
--distributed-world-size $TOTAL_GPU --distributed-port $DISTRIBUTED_PORT
end=`date +%s`
runtime=$((end-start))
