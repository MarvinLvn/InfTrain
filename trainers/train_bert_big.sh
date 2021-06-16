#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --partition=gpu_p2            # access to octo-gpus machines
#SBATCH --nodes=1                     # nombre de noeud
#SBATCH --gres=gpu:8                  # nombre de GPUs par nœud
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --exclusive

SPAN_SIZE=10
MAX_TOKENS=8192
GPU_PER_TASK=8
CPU_PER_TASK=64
TASKS_PER_NODE=1
NODES=4
TOTAL_GPU=$((GPU_PER_TASK * TASKS_PER_NODE * NODES))
DISTRIBUTED_PORT=52663
UPDATE_FREQ=$((128 / TOTAL_GPU))


TRAIN_BIN_PATH=$1
OUTPUT=deduce from train bin path if possible
NB_EPOCHS=deduce from path db

if [ -f ${PATH_CPT}/running.state ]; then
  echo "${PATH_CPT}/running.state found. Not running anything."
  exit
fi;

touch ${PATH_CPT}/running.state
python fairseq/train.py --fp16 ${TRAIN_BIN_PATH} \
      --save-dir ${OUTPUT} \
      --keep-last-epochs 1 \
      --tensorboard-logdir tensorboard \
      --train-subset train \
      --num-workers 4 \
      --task masked_lm --criterion masked_lm \
      --arch roberta_base \
      --sample-break-mode eos --tokens-per-sample 3072 --max-positions 6144 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
      --lr-scheduler polynomial_decay --lr 0.0005 --total-num-update 250000 --warmup-updates 10000 \
      --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
      --mask-multiple-length $SPAN_SIZE --mask-prob 0.5 --mask-stdev $SPAN_SIZE \
      --max-tokens $MAX_TOKENS --max-update 250000 \
      --seed 5 --log-format simple --log-interval 10 --skip-invalid-size-inputs-valid-test \
      --distributed-world-size $TOTAL_GPU --distributed-port $DISTRIBUTED_PORT


rm ${PATH_CPT}/running.state
if [ -f ${PATH_CPT}/checkpoint${NB_EPOCHS}.pt ]; then
  touch ${PATH_CPT}/done.state
fi;