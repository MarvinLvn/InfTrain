#!/bin/bash
#SBATCH --account=ank@gpu
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=10:00:00


source activate inftrain
HOME=/gpfsdswork/projects/rech/ank/ucv88ce/
export PYTHONPATH=$HOME/repos/CPC_torch:$HOME/projects/MultilingualCPC/WavAugment:$PYTHONPATH

for s in 1s 10s; do
for lang in french english; do

    out=experiments/checkpoints/cv/mono/mono_fr/eval/$lang/${s}
    mkdir -p $out
     srun python ~/repos/CPC_torch/cpc/eval/eval_ABX.py from_checkpoint experiments/checkpoints/cv/mono/mono_fr/checkpoint_42.pt /gpfsssd/scratch/rech/ank/ucv88ce/data/zerospeech/zerospeech2017_dataset/test/$lang/$s/${s}.item --speaker-level 0 /gpfsssd/scratch/rech/ank/ucv88ce/data/zerospeech/zerospeech2017_dataset/test/$lang/$s --seq_norm --strict --file_extension wav --out $out ;
done
done
