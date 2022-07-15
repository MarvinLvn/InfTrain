"""
This script computes CPC loss and accuracy on any dataset.
Example call:
python utils/validate_CPC.py --pathModel /checkpoint/marvinlvn/InfTrain/InfTrain_models/EN/3200h/00/cpc_small/checkpoint_25.pt --pathDB /private/home/marvinlvn/DATA/CPC_data/test/ABX_CV/en --file_extension .wav --pathOut /checkpoint/marvinlvn/InfTrain/InfTrain_models/EN/3200h/00/cpc_small/CommonVoiceLoss/EN/25/loss.json
"""
import argparse
import json
import os
import sys
import time

import cpc.criterion as cr
import cpc.feature_loader as fl
import cpc.utils.misc as utils
import numpy as np
import torch
from cpc.clustering.clustering import buildNewPhoneDict
from cpc.cpc_default_config import set_default_cpc_config
from cpc.criterion.research import CPCBertCriterion
from cpc.dataset import AudioBatchData, findAllSeqs
from tqdm import tqdm


def getCriterion(args, downsampling, nSpeakers, nPhones):
    dimFeatures = args.hiddenGar if not args.onEncoder else args.hiddenEncoder
    if not args.supervised:
        if args.cpc_mode == "bert":
            cpcCriterion = CPCBertCriterion(args.hiddenGar,
                                            args.hiddenEncoder,
                                            args.negativeSamplingExt)
        elif args.cpc_mode == 'none':
            cpcCriterion = cr.NoneCriterion()
            args.cluster_delay = 0
        else:
            mode = "cumNorm" if args.normMode == "cumNorm" else args.cpc_mode
            sizeInputSeq = (args.sizeWindow // downsampling)
            cpcCriterion = cr.CPCUnsupersivedCriterion(args.nPredicts,
                                                       args.hiddenGar,
                                                       args.hiddenEncoder,
                                                       args.negativeSamplingExt,
                                                       mode=mode,
                                                       rnnMode=args.rnnMode,
                                                       dropout=args.dropout,
                                                       nSpeakers=nSpeakers,
                                                       speakerEmbedding=args.speakerEmbedding,
                                                       sizeInputSeq=sizeInputSeq,
                                                       multihead_rnn=args.multihead_rnn,
                                                       transformer_pruning=args.transformer_pruning)
    elif args.pathPhone is not None:
        if not args.CTC:
            cpcCriterion = cr.PhoneCriterion(dimFeatures,
                                             nPhones, args.onEncoder,
                                             nLayers=args.nLevelsPhone)
        else:
            cpcCriterion = cr.CTCPhoneCriterion(dimFeatures,
                                                nPhones, args.onEncoder)
    else:
        cpcCriterion = cr.SpeakerCriterion(dimFeatures, nSpeakers)
    return cpcCriterion


def loadCriterion(pathCheckpoint, downsampling, nSpeakers, nPhones):
    _, _, locArgs = fl.getCheckpointData(os.path.dirname(pathCheckpoint))
    criterion = getCriterion(locArgs, downsampling, nSpeakers, nPhones)

    state_dict = torch.load(pathCheckpoint, 'cpu')

    criterion.load_state_dict(state_dict["cpcCriterion"])
    return criterion

def valStep(dataLoader,
            cpcModel,
            cpcCriterion):

    cpcCriterion.eval()
    cpcModel.eval()
    logs = {}
    cpcCriterion.eval()
    cpcModel.eval()
    iter = 0

    for step, full_data in tqdm(enumerate(dataLoader)):
        sequence, label = [x.cuda(non_blocking=True) for x in full_data]

        past, future = sequence[:, 0, ...], sequence[:, 1, ...]
        label = torch.cat([label, label])

        b = past.size(0)

        with torch.no_grad():
            combined = torch.cat([past, future], dim=0)
            c_feature, encoded_data, label = cpcModel(combined, label)
            c_feature = c_feature[:b, ...]
            encoded_data = encoded_data[b:, ...]
            label = label[:b]

            allLosses, allAcc = cpcCriterion(c_feature, encoded_data, label)

        if "locLoss_val" not in logs:
            logs["locLoss_val"] = np.zeros(allLosses.size(1))
            logs["locAcc_val"] = np.zeros(allLosses.size(1))

        iter += 1
        logs["locLoss_val"] += allLosses.mean(dim=0).cpu().numpy()
        logs["locAcc_val"] += allAcc.mean(dim=0).cpu().numpy()

    logs = utils.update_logs(logs, iter)
    logs["iter"] = iter
    utils.show_logs("Validation loss:", logs)
    return logs

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Validation')

    # Default arguments:
    parser = set_default_cpc_config(parser)

    group_db = parser.add_argument_group('Dataset')
    group_db.add_argument('--pathModel', type=str, required=True,
                          help='Path to a CPC checkpoint.')
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

    group_save = parser.add_argument_group('Save')
    group_save.add_argument('--pathCheckpoint', type=str, default=None,
                            help="Path of the output directory.")
    group_save.add_argument('--logging_step', type=int, default=1000)
    group_save.add_argument('--save_step', type=int, default=5,
                            help="Frequency (in epochs) at which a checkpoint "
                            "should be saved")

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

    print(f'CONFIG:\n{json.dumps(vars(args), indent=4, sort_keys=True)}')
    print('-' * 50)

    seqNames, speakers = findAllSeqs(args.pathDB,
                                     extension=args.file_extension,
                                     loadCache=not args.ignore_cache,
                                     speaker_level=0)

    if args.debug:
        seqNames = seqNames[:50]

    print(f'Found files: {len(seqNames)} seqs, {len(speakers)} speakers')
    print(f'Loading audio data at {args.pathDB}')
    valDataset = AudioBatchData(args.pathDB,
                            args.sizeWindow,
                            seqNames,
                            None,
                            len(speakers),
                            nProcessLoader=args.n_process_loader,
                            MAX_SIZE_LOADED=args.max_size_loaded)
    print("Dataset loaded")

    args.load = []
    cpcModel = fl.loadModel([args.pathModel])[0]
    cpcCriterion = loadCriterion(args.pathModel, cpcModel.gEncoder.DOWNSAMPLING,
                                 len(speakers), None)
    cpcModel.cuda()
    cpcCriterion.cuda()
    cpcModel = torch.nn.DataParallel(cpcModel,
                                     device_ids=range(args.nGPU)).cuda()
    cpcCriterion = torch.nn.DataParallel(cpcCriterion,
                                         device_ids=range(args.nGPU)).cuda()

    valLoader = valDataset.getDataLoader(batchSize, 'sequential', False,
                                         numWorkers=0)
    locLogsVal = valStep(valLoader, cpcModel, cpcCriterion)
    scores = {'loss': locLogsVal['locLoss_val'].mean(),
              'acc': locLogsVal['locAcc_val'].mean(),
              'loss_per_timestep': locLogsVal['locLoss_val'].tolist(),
              'acc_per_timestep': locLogsVal['locAcc_val'].tolist()}
    print(scores)
    os.makedirs(os.path.dirname(args.pathOut), exist_ok=True)

    with open(args.pathOut, 'w') as fout:
        json.dump(scores, fout, indent=2)
    print("Loss: %.3f" % scores['loss'])
    print("Acc: %.3f" % scores['acc'])


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = sys.argv[1:]
    main(args)
