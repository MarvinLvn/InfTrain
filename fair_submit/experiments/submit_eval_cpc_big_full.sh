DATA_PATH=/private/home/marvinlvn/DATA/CPC_data/train/InfTrain/dataset/wav
LANGUAGES=(EN FR)

cd ..
for LANGUAGE in ${LANGUAGES[*]}; do
  sbatch -o eval_cpc_big_${LANGUAGE}_full.txt evaluators/evaluate_cpc_ls.sh $DATA_PATH/$LANGUAGE
done;
