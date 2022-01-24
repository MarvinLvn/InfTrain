import os
from time import time
import torch
import torchaudio
from cpc.feature_loader import loadModel, getCheckpointData
from cpc.train import getCriterion
import argparse
from cpc.dataset import findAllSeqs
from pathlib import Path
from tqdm import tqdm
import sys
import pandas as pd
import numpy as np

def loadCriterion(pathCheckpoint, downsampling, nSpeakers, nPhones):
    _, _, locArgs = getCheckpointData(os.path.dirname(pathCheckpoint))

    if 'speakerEmbedding' in locArgs and locArgs.speakerEmbedding == None:
        locArgs.speakerEmbedding = 0
    locArgs.size_speaker_emb=0

    criterion = getCriterion(locArgs, downsampling, nSpeakers, nPhones)

    state_dict = torch.load(pathCheckpoint, 'cpu')

    criterion.load_state_dict(state_dict["cpcCriterion"])
    return criterion


def getPositiveSamples(encodedData, nPredicts=12):
    batchSize, nNegativeExt, dimEncoded = encodedData.size()
    outputs = []

    for k in range(1, nPredicts + 1):
        # Positive samples
        if k < nPredicts:
            posSeq = encodedData[:, k:-(nPredicts - k)]
        else:
            posSeq = encodedData[:, k:]
        posSeq = posSeq.view(batchSize, 1, posSeq.size(1), dimEncoded)
        outputs.append(posSeq)

    return outputs


def compute_score_CPC(seqPath,
                      cpcModel,
                      cpcCriterion,
                      speakerLabel=0,
                      nTemporal=12,
                      logits_scaling=1,
                      reduce_method='sum',
                      prob_estimator='negative_sampling',
                      n_negative_sampling=64
                      ):
    '''
    Comment on some useful args:
        logits_scaling:  put this high to avoid having 0. log proba for near temporal steps when
                         using sigmoid, but it seems that 1 (default) gives the best results
        reduce_method:  'sum' seems to work best
        prob_estimator:  using 'sigmoid' is faster as we don't need to compute negative samples,
                         but using 'negative_sampling' seems to have better results as this is
                         the way the CPC model is trained (however this will make the scores varying)
   n_negative_sampling:  number of negative sample, default to 8
    '''
    assert reduce_method in ['sum', 'mean']
    assert prob_estimator in ['sigmoid', 'negative_sampling', 'loss']
    with torch.no_grad():
        # Read the input signals
        seq = torchaudio.load(seqPath)[0]  # 1 x frames
        seq = seq[:, :].view(1, 1, -1).cuda()  # 1 x 1 x frames

        # Read CPC features
        cpcModel.gAR.hidden = None
        cFeature, encodedData, label = cpcModel(seq, label=None)

        ## cFeature: 1 x T x D_feat
        ## encodedData: 1 x T x D_enc

        # Prepare CPC features for criterion
        batchSize, seqSize, _ = cFeature.size()
        windowSize = seqSize - cpcCriterion.nPredicts  # T - 12
        cFeature = cFeature[:, :windowSize]  # 1 x (T - 12) x D_feats

        # Get positive encoded samples
        if prob_estimator == 'negative_sampling' or prob_estimator == 'loss':
            if n_negative_sampling is not None:
                cpcCriterion.negativeSamplingExt = n_negative_sampling

            sampledData, labelLoss = cpcCriterion.sampleClean(encodedData,
                                                              windowSize)  # 12 x 1 x (1 + n_negative_sampling) x (T - 12) x D_enc
        else:
            sampledData = getPositiveSamples(encodedData, cpcCriterion.nPredicts)  # 12 x 1 x 1 x (T - 12) x D_enc

        # Speaker embeddings
        if getattr(cpcCriterion, 'speakerEmb', None) is not None:
            label = torch.tensor(speakerLabel).cuda()
            l_ = label.view(batchSize, 1).expand(batchSize, windowSize)  # 1 x (T - 12)
            embeddedSpeaker = cpcCriterion.speakerEmb(l_)  # 1 x (T - 12) x D_spkemb
            cFeature = torch.cat([cFeature, embeddedSpeaker], dim=2)  # 1 x (T - 12) x (D_feat+D_spkemb)

        # Compute the criterion outputs
        predictions = cpcCriterion.wPrediction(cFeature, sampledData)  # 12 x 1 x 1 x (T - 12)

        # Compute the pseudo log-probas
        lp_score = 0.
        outLosses = [0 for x in range(nTemporal)]
        outAcc = [0 for x in range(nTemporal)]
        for k, outputs in enumerate(predictions[:nTemporal]):
            if prob_estimator == 'sigmoid':
                logits = outputs[0] / logits_scaling
                logits = logits.sigmoid()
            elif prob_estimator == 'negative_sampling':
                logits = outputs[0] / logits_scaling
                logits = logits.softmax(0)
            elif prob_estimator == 'loss':
                outputs = outputs.permute(0, 2, 1)
                outputs = outputs.contiguous().view(-1, outputs.size(2))
                lossK = cpcCriterion.lossCriterion(outputs, labelLoss)

                outLosses[k] += lossK.view(1, -1)
                _, predsIndex = outputs.max(1)
                outAcc[k] += torch.sum(predsIndex == labelLoss).float().view(1, -1).item()

            # if reduce_method == 'sum':
            #     lp_score += logits[0].log().sum()
            # elif reduce_method == 'mean':
            #     lp_score += logits[0].log().mean()
            #lp_score /= nTemporal

            # logits = outputs[0] / logits_scaling
            # if logits.size(0) == 1:
            #     logits = logits.sigmoid()
            # else:
            #     logits = logits.softmax(0)
            # if reduce_method == 'sum':
            #     lp_score += logits[0].log().sum()
            # elif reduce_method == 'mean':
            #     lp_score += logits[0].log().mean()
            # lp_score /= nTemporal
        outLosses = torch.FloatTensor(outLosses).cpu() / (windowSize * batchSize)
        outAcc = torch.FloatTensor(outAcc).cpu() / (windowSize * batchSize)

    return outLosses, outAcc


def main(argv):
    parser = argparse.ArgumentParser(
        description='Given a folder containg audio files, will compute pseudo-probabilities of a CPC model.')
    parser.add_argument('--cpt', type=str, required=True,
                        help='Path to a CPC checkpoint (.pt) or '
                             'a .csv file that contains CPC pseudo-probabilities.')
    parser.add_argument('--out', type=str, required=True,
                        help='Path where the output file will be stored (.txt)')
    parser.add_argument('--db', type=str, required=True,
                        help='Path to the folder containing the audio files.')
    parser.add_argument('--extension', type=str, default='.wav',
                        help='Extension of the audio files (default to .wav).')
    parser.add_argument('--reduce_method', type=str, choices=['sum', 'mean'], default='sum')
    parser.add_argument('--prob_estimator', type=str, choices=['sigmoid', 'negative_sampling', 'loss'], default='negative_sampling')
    parser.add_argument('--debug', action='store_true',
                        help='If activated, will consider only first 5 audio files.')
    args = parser.parse_args(argv)

    # Load CPC model
    cpcModel = loadModel([args.cpt])[0].cuda()
    cpcModel.gAR.keepHidden = True
    cpcModel.eval()

    # Load CPC criterion
    cpcCriterion = loadCriterion(args.cpt, cpcModel.gEncoder.DOWNSAMPLING, 0, None).cuda()
    cpcCriterion.eval()
    print('CPC model and criterion loaded!')

    seq_list, _ = findAllSeqs(args.db, speaker_level=0, extension=args.extension)
    seq_list = [(str(Path(x).stem), str(Path(args.db) / x))
                for (_, x) in seq_list]
    if args.debug:
        seq_list = seq_list[:50]
    print(f'{len(seq_list)} files found!')

    out = []
    for i, file in tqdm(enumerate(seq_list)):
        outLoss, outAcc = compute_score_CPC(seqPath=file[1],
                               cpcModel=cpcModel,
                               cpcCriterion=cpcCriterion,
                               reduce_method=args.reduce_method,
                               prob_estimator=args.prob_estimator)
        basename = os.path.basename(file[1]).replace(args.extension, '')
        out.append([basename, outAcc.mean().item()])

    out = pd.DataFrame(out)
    out.columns = ["filename", "prob"]
    out.to_csv(args.out, sep=" ", header=False, index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
