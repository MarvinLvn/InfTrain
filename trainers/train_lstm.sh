#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters. Please respect :"
  echo "./train_lstm.sh /path/to/database/containing/wav/files"
  echo "Example :"
  echo "./train_lstm.sh /gpfsscratch/rech/cfs/commun/families/EN/50h/00"
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
python "${BASELINE_SCRIPTS}/quantize_audio.py" "${PATH_KMEANS}" "${TRAIN_SET}" "${OUTPUT}" --file_extension $FILE_EXT --resume

# check nb files
NB_FILES=$(find $TRAIN_SET -name "*$FILE_EXT" | wc -l)
NB_LINES=$(cat $OUTPUT_LOCATION/quantized_train/quantized_outputs.txt | wc -l)

echo "Found $NB_FILES files".
echo "$(($NB_LINES+1)) sequences have been quantized"

if [ $(($NB_LINES+1)) -ne $NB_FILES ]; then
  echo "Wrong number of quantized sequences. Stopped."
  exit
fi;

# + convert format
cat $OUTPUT_LOCATION/quantized_train/quantized_outputs.txt | awk '{print $2}' | sed 's/,/ /g' > $OUTPUT_LOCATION/quantized_train/quantized_outputs_2.txt
# + split train/val/test
python ${BASELINE_SCRIPTS}/split_train_val_test_lm.py --input_file $OUTPUT_LOCATION/quantized_train/quantized_outputs_2.txt

# 2) Fairseq preprocess
fairseq-preprocess --only-source \
      --trainpref $OUTPUT_LOCATION/quantized_train/fairseq_train.txt \
      --validpref $OUTPUT_LOCATION/quantized_train/fairseq_val.txt \
      --testpref $OUTPUT_LOCATION/quantized_train/fairseq_test.txt \
      --destdir $OUTPUT_LOCATION/fairseq_bin_data \
      --workers 20


# 3) Train models
MODEL_OUTPUT=${MODELS_PATH}/${LANGUAGE}/${SIZE}/${SHARE}/lstm
python ${FAIRSEQ_SCRIPTS}/train.py --fp16 $OUTPUT_LOCATION/fairseq_bin_data \
      --task language_modeling \
      --save-dir ${MODEL_OUTPUT} \
      --keep-last-epochs 2 \
      --tensorboard-logdir tensorboard \
      --arch lstm_lm \
      --decoder-embed-dim 200 --decoder-hidden-size 1024 --decoder-layers 3 \
      --decoder-out-embed-dim 200 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
      --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 1000 --warmup-init-lr 1e-07 \
      --dropout 0.1 --weight-decay 0.01 \
      --sample-break-mode none --tokens-per-sample 2048 \
      --max-tokens 163840 --update-freq 1 --max-update 100000
