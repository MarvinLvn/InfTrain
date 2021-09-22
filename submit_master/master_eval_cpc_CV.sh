#!/bin/bash
#SBATCH --account=ank@gpu
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread

source activate inftrain
HOME=/gpfsdswork/projects/rech/ank/ucv88ce
export PYTHONPATH=$HOME/repos/CPC_torch:$HOME/projects/MultilingualCPC/WavAugment:$PYTHONPATH

#model=mono/mono_en
model=mix/en-fr_A+B

BEST_EPOCH_PY=/gpfsdswork/projects/rech/ank/ucv88ce/projects/MultilingualCPC/utils/best_val_epoch.py

ZS17="/gpfsssd/scratch/rech/ank/ucv88ce/data/zerospeech/zerospeech2017_dataset/test"
for s in 1s 1s_easyfilter; do
    
for lang in french english; do
    echo ${ZS17}/$lang/$s/${s}.item
    if [ -f ${ZS17}/$lang/$s/${s}.item ]; then
        
        BEST_EPOCH="$(python "$BEST_EPOCH_PY" --output-id --model_path experiments/checkpoints/cv/$model)"
        echo "Best epoch is $BEST_EPOCH for $model"
        out=experiments/checkpoints/cv/$model/ABX/$BEST_EPOCH/$lang/${s}
        mkdir -p $out
        if [ ! -f $out/ABX_scores.json ]; then

            echo "Running eval in $out"
            srun python ~/repos/CPC_torch/cpc/eval/eval_ABX.py from_checkpoint experiments/checkpoints/cv/${model}/checkpoint_${BEST_EPOCH}.pt /gpfsssd/scratch/rech/ank/ucv88ce/data/zerospeech/zerospeech2017_dataset/test/$lang/$s/${s}.item --speaker-level 0 /gpfsssd/scratch/rech/ank/ucv88ce/data/zerospeech/zerospeech2017_dataset/test/$lang/$s --seq_norm --strict --file_extension wav  --out $out ;
        fi
    fi
done
done
