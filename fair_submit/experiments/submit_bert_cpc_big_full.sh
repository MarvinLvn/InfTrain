DATA_PATH=/private/home/marvinlvn/DATA/CPC_data/train/InfTrain/dataset/wav
LANGUAGES=(EN FR)

cd ..
for LANGUAGE in ${LANGUAGES[*]}; do
  sbatch -o bert_cpc_big_${LANGUAGE}_full.txt trainers/train_bert_big.sh $DATA_PATH/$LANGUAGE
done;
