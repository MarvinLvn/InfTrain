DATA_PATH=/private/home/marvinlvn/DATA/CPC_data/train/InfTrain/dataset/wav
LANGUAGES=(EN FR)

cd ..
for LANGUAGE in ${LANGUAGES[*]}; do
  sbatch -o cpc_small_${LANGUAGE}_full.txt trainers/train_cpc_small.sh $DATA_PATH/$LANGUAGE
done;
