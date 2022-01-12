LANGUAGES=(en fr)
SHARES=(64 32 16 8 4 2 1)

for LANGUAGE in ${LANGUAGES[*]}; do
  for SHARE in ${SHARES[*]}; do
    sbatch submit_eval_hadrien_lexical.sh lstm $LANGUAGE testset_${SHARE}
  done;
done;