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

#model=mono/mono_fr
#model=mix/en-fr_A+B

BEST_EPOCH_PY=/gpfsdswork/projects/rech/ank/ucv88ce/projects/MultilingualCPC/utils/best_val_epoch.py

CV21="/gpfsscratch/rech/cfs/commun/cv21_ABX/raw_dataset"

#BEST_EPOCH="$(python "$BEST_EPOCH_PY" --output-id --model_path experiments/checkpoints/cv/$model)"

BEST_EPOCH=20

for lang in fr en; do
    #for s in ${lang} ${lang}_pos; do
    for s in ${lang}; do
        

        echo "Best epoch is $BEST_EPOCH for $model"
        PATH_OUT=experiments/checkpoints/cv/$model/ABX/$BEST_EPOCH/cv_test/$lang/${s}
        mkdir -p $PATH_OUT
        
        PATH_ITEM_FILE="$CV21/${lang}/${s}.item"
        LANG_DATASET="${CV21}/${lang}" 
 
        if [ ! -f $PATH_OUT/ABX_scores.json ]; then

            echo "Running eval in $out"
            srun python ~/repos/CPC_torch/cpc/eval/eval_ABX.py from_checkpoint experiments/checkpoints/cv/${model}/checkpoint_${BEST_EPOCH}.pt $PATH_ITEM_FILE --speaker-level 0 $LANG_DATASET --seq_norm --strict --file_extension wav  --out $PATH_OUT ;
        fi
done
done
