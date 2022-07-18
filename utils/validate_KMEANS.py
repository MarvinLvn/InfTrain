"""
This script computes CPC loss and accuracy on any dataset.
Example call:
python utils/validate_KMEANS.py \
    --pathModel /checkpoint/marvinlvn/InfTrain/InfTrain_models/EN/3200h/00/kmeans/checkpoint_last.pt \
    --pathDB/private/home/marvinlvn/DATA/CPC_data/test/ABX_CV/en --file_extension .wav \
    --pathOut /checkpoint/marvinlvn/InfTrain/InfTrain_models/EN/3200h/00/cpc_small/CommonVoiceLoss/EN/25/loss.json
"""
from pathlib import Path
from cpc.feature_loader import (
    FeatureModule,
    loadModel,
)
from cpc.clustering.clustering import loadClusterModule
from utils.utils_functions import readArgs
import argparse
import json
import os
import sys

import cpc.criterion as cr
import torch
from cpc.criterion.research import CPCBertCriterion
from cpc.dataset import AudioBatchData, findAllSeqs
from tqdm import tqdm

def valStep(
        dataLoader,
        clusterModule,
        featureMaker,
):
    print("Computing CPC features...")
    outData = []
    for index, item in enumerate(dataLoader):

        with torch.no_grad():
            features = featureMaker(item).cpu()

        N, S, C = features.size()
        outData.append(features.contiguous().view(N * S, C))
    print("Done")

    Ck = clusterModule.module.Ck
    k = Ck.size(1)
    D = Ck.size(2)
    scores = torch.zeros(k).cuda()
    n_items = torch.zeros(k).cuda()
    with torch.no_grad():
        for index, data in tqdm(enumerate(dataLoader)):
            cFeatures = featureMaker(data).contiguous().view(-1, 1, D)
            qFeatures = clusterModule(cFeatures)
            assigned_clusters = torch.argmin(qFeatures, dim=-1).view(-1)
            distance_to_cluster = torch.min(qFeatures, dim=-1).values.view(-1)

            # number of points assigned to each cluster
            n_items += torch.cat([(assigned_clusters == p).sum(dim=0, keepdim=True) for p in range(k)], dim=0)
            # distance of points to their assigned cluster
            scores += torch.cat([distance_to_cluster[(assigned_clusters == p).view(-1)].sum(dim=0, keepdim=True) for p in range(k)], dim=0)

    score_per_cluster = torch.nan_to_num(scores / n_items, nan=0).cpu().numpy()
    return score_per_cluster, score_per_cluster.mean().item()

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Validation')

    group_db = parser.add_argument_group('Dataset')
    group_db.add_argument('--pathModel', type=str, required=True,
                          help='Path to a KMEANS checkpoint.')
    group_db.add_argument('--pathDB', type=str, required=True,
                          help='Path to the directory containing the '
                          'data.')
    group_db.add_argument('--pathOut', type=str, required=True,
                          help='Where to score the scores.')
    group_db.add_argument('--file_extension', type=str, default=".wav",
                          help="Extension of the audio files in the dataset.")
    group_db.add_argument('--n_process_loader', type=int, default=8,
                          help='Number of processes to call to load the '
                          'dataset')
    group_db.add_argument('--ignore_cache', action='store_true',
                          help='Activate if the dataset has been modified '
                          'since the last training session.')
    group_db.add_argument('--max_size_loaded', type=int, default=4000000000,
                          help='Maximal amount of data (in byte) a dataset '
                          'can hold in memory at any given time')

    group_gpu = parser.add_argument_group('GPUs')
    group_gpu.add_argument('--nGPU', type=int, default=-1,
                           help="Number of GPU to use (default: use all "
                           "available GPUs)")
    group_gpu.add_argument('--batchSizeGPU', type=int, default=8,
                           help='Number of batches per GPU.')
    parser.add_argument('--debug', action='store_true',
                        help="Load only a very small amount of files for "
                        "debugging purposes.")
    args = parser.parse_args(argv)
    args.pathModel = os.path.abspath(args.pathModel)
    if not args.pathOut[-5:] == '.json':
        raise ValueError("pathOut should points to a .json file.")

    if args.nGPU < 0:
        args.nGPU = torch.cuda.device_count()
    assert args.nGPU <= torch.cuda.device_count(),\
        f"number of GPU asked: {args.nGPU}," \
        f"number GPU detected: {torch.cuda.device_count()}"
    print(f"Let's use {args.nGPU} GPUs!")
    return args

def main(argv):
    args = parseArgs(argv)
    batchSize = args.nGPU * args.batchSizeGPU


    # Load model
    assert args.pathModel.endswith(".pt")
    clustering_args = readArgs(Path(args.pathModel).parent / 'args.json')
    print("")
    print(
        f"Clustering args:\n{json.dumps(vars(clustering_args), indent=4, sort_keys=True)}"
    )
    print("-" * 50)

    clusterModule = loadClusterModule(args.pathModel)
    clusterModule.eval()
    clusterModule.cuda()

    print("")
    print("Loading CPC FeatureMaker")
    if "level_gru" in vars(clustering_args) and clustering_args.level_gru is not None:
        updateConfig = argparse.Namespace(nLevelsGRU=clustering_args.level_gru)
    else:
        updateConfig = None
    model = loadModel([clustering_args.pathCheckpoint], updateConfig=updateConfig)[0]
    ## If we don't apply batch implementation, we can set LSTM model to keep hidden units
    ## making the quality of the quantized units better
    # if args.nobatch:
    #     model.gAR.keepHidden = True
    featureMaker = FeatureModule(model, clustering_args.encoder_layer)
    featureMaker.eval()
    featureMaker.cuda()
    print("CPC FeatureMaker loaded!")

    # Load audio
    seqNames, speakers = findAllSeqs(args.pathDB,
                                     extension=args.file_extension,
                                     loadCache=not args.ignore_cache,
                                     speaker_level=0)

    if args.debug:
        seqNames = seqNames[:50]

    print(f'Found files: {len(seqNames)} seqs, {len(speakers)} speakers')
    print(f'Loading audio data at {args.pathDB}')
    valDataset = AudioBatchData(args.pathDB,
                                clustering_args.sizeWindow,
                                seqNames,
                                None,
                                len(speakers),
                                nProcessLoader=args.n_process_loader,
                                MAX_SIZE_LOADED=args.max_size_loaded)
    print("Dataset loaded")


    clusterModule = torch.nn.DataParallel(clusterModule, device_ids=range(args.nGPU)).cuda()
    featureMaker = torch.nn.DataParallel(featureMaker, device_ids=range(args.nGPU)).cuda()

    valLoader = valDataset.getDataLoader(batchSize, 'sequential', False,
                                         numWorkers=0)
    score_per_cluster, score = valStep(valLoader, clusterModule, featureMaker)
    scores = {'loss': score,
              'loss_per_cluster': score_per_cluster.tolist()}
    print(scores)

    os.makedirs(os.path.dirname(args.pathOut), exist_ok=True)
    with open(args.pathOut, 'w') as fout:
        json.dump(scores, fout, indent=2)
    print("Loss: %.3f" % score)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    args = sys.argv[1:]
    main(args)
