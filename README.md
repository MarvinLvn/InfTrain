# Installation

Please follow instructions at https://github.com/bootphon/zerospeech2021_baseline
In the following, we'll assume that experiments will be performed on the [Jean Zay cluster](http://www.idris.fr/annonces/annonce-jean-zay-eng.html). 

# Structure

All files generated by the experiments (training and evaluation) should be stored under `/gpfsscratch/rech/cfs/commun/experiments`.
Files should be organized as follows : 

```
|English
│ 
└─── 50h
     │ 
     └─── 00
            └─── cpc_<small|big>
                 | kmeans_50 
                 | <lstm|bert_large>
                 | evaluation
         |01
         |...
         |63
     |100h/
         |00
         |...
         |31
     |200h/
     |400h/
     |800h/
     |1600h/
     |3200h/
         |00
|French
│ 
└─── same structure
```

The `trainers` folder contain scripts to train the different models (CPC, K-means, BERT, lstm).
The `experiments` folder contain scripts to generate the different experiments (training set, model, model parameters). These scripts should generate the configuration of each experiment in a `.txt` file whose each line will be submitted via a [slurm job array](http://www.idris.fr/jean-zay/cpu/jean-zay-cpu-exec_jobarray.html)

# Installation

```bash
module load sox
conda env create -f environment.yml && conda activate inftrain
git clone https://github.com/MarvinLvn/CPC_torch.git
```

To train models, you must install the following dependencies : 

- [CPC models and K-means](https://github.com/fairinternal/CPC_torch) (FAIR only)
- [Language models](https://github.com/pytorch/fairseq)

Please refer to [this git repo](https://github.com/bootphon/zerospeech2021_baseline) for instructions about how to train the model

To evaluate models, you must install the [ZeroSpeech 2021 repo](https://github.com/bootphon/zerospeech2021)

# How to connect to the account ?

From flores :

```angular2html
ssh uow84uh@jean-zay.idris.fr

# Load right project (to have access to inftrain conda env)
cd utils 
source cfs_proj.sh
```

# Running experiments

All experiments will be run on Marvin's Jean Zay account. This git repo can be found under `/gpfsscratch/rech/cfs/uow84uh/InfTrain` with pre-installed dependencies.
To run experiments, first type :

```bash
cd experiments
./generate_study.sh
```

This will create experiment files in the `experiment_txt` folder. 
There's one experiment file for each model, and each line of an experiment file contains the path to the training set.
The information of which model needs to be trained is automatically deduced from the training set path.

Once you generated experiment files, you can check their content and then run :

```bash
# To submit CPC small models
sbatch submit_cpc_small.sh

# To submit CPC big models
sbatch submit_cpc_big.sh
```

None of the scripts to train k-means and language models work now. 
Those should be finished and thoroughly checked before running anything.

# Submit training individually :

CPC small : 

```bash
sbatch -o my_log_cpc_small_srun.txt trainers/train_cpc_small.sh /gpfsscratch/rech/cfs/commun/families/EN/50h/00
```

CPC big :

```bash
sbatch -o my_log_cpc_big.txt trainers/train_cpc_big.sh /gpfsscratch/rech/cfs/commun/families/EN/3200h/00
```

# How it works ?

Each time a model is trained, let's say in `EN/50h/00/cpc_small`, a file `running.state` is created.
If the model reaches its planned number of epochs, this file will be replaced by `done.state`.
The `generate_study.sh` script will look for the presence of either one file or the other to decide if a given model needs to be (re-)trained.

However, one should be careful. This system hasn't been thoroughly checked. Not 100% clear to me what happens when there's a memory issue for instance.
Will the `running.state` file be removed ? If no, we'll have to remove them manually so that `generate_study.sh` knows which models need to be retrained.

# What needs to be done ?

- Check CPC models are running / converging (plot validation loss for different training duration)
- Create a python script or jupyter notebook that shows the progress of the study : how many CPC models have been fully trained ? K-means ? Languages models ? Etc
- Prepare submission scripts for all the metrics : ABX, sSIMI, sBLIMP, sWUGGY (see with Nick)
- Finish submission scripts to train k-means and language models.
- Create submission scripts to extract discrete-representation
