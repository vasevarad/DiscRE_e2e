import sys
import csv
import pickle
import pandas as pd
from utils.SenseLabeller import SenseLabeller
import utils.PDTBAppendixSenses as PDTBas
import sys
import numpy as np
import torch
import torch.autograd as autograd
from collections import OrderedDict
from copy import deepcopy

def genTrainWeights(sl, disCon):
    explicitInstances={'class':[],'type':[],'subtype':[]} # list of (label,weight) pairs
    implicitInstances = {'class': [], 'type': [], 'subtype': []}  # list of (label,weight) pairs

    if disCon in sl.weightDict['explicit']:
        for idx in np.nonzero(sl.weightDict['explicit'][disCon]['class'])[0]:
            label=torch.zeros(4)
            label[idx]=1
            explicitInstances['class'].append((label,sl.weightDict['explicit'][disCon]['class'][idx]))
        for idx in np.nonzero(sl.weightDict['explicit'][disCon]['type'])[0]:
            label = torch.zeros(16)
            label[idx] = 1
            explicitInstances['type'].append((label,sl.weightDict['explicit'][disCon]['type'][idx]))
        for idx in np.nonzero(sl.weightDict['explicit'][disCon]['subtype'])[0]:
            label = torch.zeros(25)
            label[idx] = 1
            explicitInstances['subtype'].append((label, sl.weightDict['explicit'][disCon]['subtype'][idx]))
    else:
        explicitInstances=None

    if disCon in sl.weightDict['implicit']:
        for idx in np.nonzero(sl.weightDict['implicit'][disCon]['class'])[0]:
            label = torch.zeros(4)
            label[idx] = 1
            implicitInstances['class'].append((label,sl.weightDict['implicit'][disCon]['class'][idx]))
        for idx in np.nonzero(sl.weightDict['implicit'][disCon]['type'])[0]:
            label = torch.zeros(16)
            label[idx] = 1
            implicitInstances['type'].append((label,sl.weightDict['implicit'][disCon]['type'][idx]))
        for idx in np.nonzero(sl.weightDict['implicit'][disCon]['subtype'])[0]:
            label = torch.zeros(25)
            label[idx] = 1
            implicitInstances['subtype'].append((label, sl.weightDict['implicit'][disCon]['subtype'][idx]))
    else:
        implicitInstances=None


    return explicitInstances,implicitInstances


def make_discre_input(args_file, args_meta_file):
    print("Making input files for DiscRE model ... ")
    args_data=pd.read_csv(args_file)
    args_meta_data=pd.read_csv(args_meta_file)

    explicitDict=PDTBas.explicitConnectiveRelationDict
    implicitDict=PDTBas.implicitConnectiveRelationDict

    sl=SenseLabeller()

    # arg as dict {tweet_id: [arg_0, arg_1, ..], ...}
    # meta as dict {tweet_id:[(train_1_0, case, disCon, Arg1_idx, Arg2_idx),(train_1_1, case, disCon, Arg1_idx, Arg2_idx),...], ...}
    argDict={}
    argMetaDict=OrderedDict()
    labelDict={}
    impTrDict={}


    for idx,row in args_data.iterrows():
        try:
            argDict[row['tweet_id']].append(str(row['message']).split())
        except KeyError:
            argDict[row['tweet_id']]=[str(row['message']).split()]


    for idx,row in args_meta_data.iterrows():
        if row['Discourse_Connective'] in explicitDict or row['Discourse_Connective'] in implicitDict:
            arg1Idx=int(str(row['Arg1_id']).split('-')[1])
            arg2Idx=int(str(row['Arg2_id']).split('-')[1])
            try:
                argMetaDict[row['tweet_id']].append((row['trainer_id'],row['Case_Type'],row['Discourse_Connective'],arg1Idx,arg2Idx))
            except KeyError:
                argMetaDict[row['tweet_id']]=[(row['trainer_id'],row['Case_Type'],row['Discourse_Connective'],arg1Idx,arg2Idx)]
            explicitInstances, implicitInstances=genTrainWeights(sl, row['Discourse_Connective'])
            labelDict[row['trainer_id']]={'explicit':explicitInstances,'implicit':implicitInstances}
            wordSeq=deepcopy(argDict[row['tweet_id']])
            wordSeq[arg2Idx]=wordSeq[arg2Idx][len(str(row['Discourse_Connective']).split()):]
            impTrDict[row['trainer_id']]=wordSeq
    pickle.dump(argDict,open(args_file[:-4]+'.dict','wb'))
    pickle.dump(argMetaDict,open(args_meta_file[:-4]+'.odict','wb'))
    pickle.dump(labelDict,open(args_meta_file[:-4]+'_labels.dict','wb'))
    pickle.dump(impTrDict,open(args_meta_file[:-4]+'_implicit_word_seqs.dict','wb'))
    print("... Done")
    return args_file[:-4]+'.dict'