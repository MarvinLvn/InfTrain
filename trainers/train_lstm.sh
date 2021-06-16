#!/bin/bash
#SBATCH --account=cfs@gpu
#SBATCH --partition=gpu_p2            # access to octo-gpus machines
#SBATCH --nodes=1                     # nombre de noeud
#SBATCH --gres=gpu:8                  # nombre de GPUs par nœud
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --exclusive

TRAIN_BIN_PATH=$1
OUTPUT=deduce from train bin path if possible

python fairseq/train.py --fp16 ${TRAIN_BIN_PATH} \
      --task language_modeling \
      --save-dir ${OUTPUT} \
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