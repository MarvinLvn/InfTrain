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
     └─── share0
               └─── cpc_<small|big>
                   | kmeans_<K> 
                   | <lstm|bert_small|bert_large>
                   | evaluation
         |share1
         |...
         |share63
     |100h/
         |share0
         |...
         |share31
     |200h/
     |400h/
     |800h/
     |1600h/
     |3200h/
         |share1
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

Once you generated experiment files, you can train CPC models by running :

```bash

```