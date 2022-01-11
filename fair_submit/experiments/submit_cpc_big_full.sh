DATA_PATH=/private/home/marvinlvn/DATA/CPC_data/train/InfTrain/dataset/wav
LANGUAGES=(EN FR)

cd ..
for LANGUAGE in ${LANGUAGES[*]}; do
  sbatch -o cpc_big_${LANGUAGE}_full.txt --constraint volta32gb trainers/train_cpc_big.sh $DATA_PATH/$LANGUAGE
done;
