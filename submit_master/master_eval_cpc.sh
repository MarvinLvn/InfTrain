#!/usr/bin/env bash

FAM_PATH=/gpfsssd/scratch/rech/cfs/commun/families

for x in 00 $(seq -w 31); do

    for lan in EN+FR EN FR; do

        fam=${FAM_PATH}/${lan}/200h/${x}

        if [ ! -f $fam/ABX/*/english/ABX_args.json ] && [ ! -f $fam/ABX/*/french/ABX_args.json ]; then
            sbatch ./evaluators/evaluate_cpc.sh ${fam};
        fi
        
    done;



    for lan in EN FR; do

        fam=${FAM_PATH}/${lan}/100h/${x}

        if [ ! -f $fam/ABX/*/english/ABX_args.json ] && [ ! -f $fam/ABX/*/french/ABX_args.json ]; then
            sbatch ./evaluators/evaluate_cpc.sh ${fam};
        fi
    done;

    for lan in EN+FR EN FR; do

        fam=${FAM_PATH}/${lan}/400h/${x}

        if [ ! -f $fam/ABX/*/english/ABX_args.json ] && [ ! -f $fam/ABX/*/french/ABX_args.json ]; then
            sbatch ./evaluators/evaluate_cpc.sh ${fam};
        fi
        
    done;
    
    
done 
