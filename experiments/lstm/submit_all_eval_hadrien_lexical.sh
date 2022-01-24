#LANGUAGES=(en fr)
#SHARES=(64 32 16 8 4 2 1)
LANGUAGES=(en fr)
SHARES=(64 1)
SBMS=(none complete eos)

for LANGUAGE in ${LANGUAGES[*]}; do
  for SHARE in ${SHARES[*]}; do
    for SBM in ${SBMS[*]}; do
      sbatch submit_eval_hadrien_lexical.sh bert_sbm_$SBM $LANGUAGE testset_${SHARE}
    done;
  done;
done;
