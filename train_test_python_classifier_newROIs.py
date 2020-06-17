# purpose: read in category separation for each run, save for all runs


import os
import glob
import argparse
import numpy as np  # type: ignore
import sys
# Add current working dir so main can be run from the top level rtAttenPenn directory
sys.path.append(os.getcwd())
import rtfMRI.utils as utils
import rtfMRI.ValidationUtils as vutils
from rtfMRI.RtfMRIClient import loadConfigFile
from rtfMRI.Errors import ValidationError
from rtAtten.RtAttenModel import getSubjectDayDir
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from rtfMRI.StructDict import StructDict, MatlabStructDict
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import scipy
from rtAtten.Test_L2_RLR_realtime import Test_L2_RLR_realtime
from sklearn import metrics
# run for each subject after subject completes all days
import sys

subjectNum=np.int(sys.argv[1]) 
ROI = sys.argv[2]

ndays = 3
auc_score = np.zeros((8,ndays)) # save larger to fit all days in
RT_cs = np.zeros((8,ndays))
nTRTest = 100
RT_cs_timecourse = np.zeros((8,nTRTest,ndays))
dataPath = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo/'
subjectDir =  dataPath + '/' + 'subject' + str(subjectNum) 
print(subjectNum)
for d in np.arange(ndays):
    print(d)
    subjectDay = d + 1
    configFile = dataPath + 'subject' + str(subjectNum) + '/usedscripts/PennCfg_Day' + str(subjectDay) + '.toml'
    cfg = loadConfigFile(configFile)
    subjectDayDir = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo/subject' + str(cfg.session.subjectNum) + '/day' + str(cfg.session.subjectDay)
    pyDataDir = subjectDayDir
    # load ROI indices
    ROI_fn = subjectDayDir + '/' + ROI + '_' + str(subjectNum) + '_' + str(subjectDay) + '.mat'
    ROI_dict = scipy.io.loadmat(ROI_fn)
    ROI_indices = ROI_dict['indices'][0]
    if subjectDay == 1:
        nRuns = 7
        if subjectNum == 106:
            nRuns = 6
    elif subjectDay == 2:
        nRuns = 9
    elif subjectDay == 3:
        nRuns = 8
    #nruns = len(cfg.session.Runs) - 1
    for runId in np.arange(1,nRuns):
        print(runId)
        runDir = 'run'+str(runId)+'/'
        pyModelFn = utils.findNewestFile(pyDataDir, 'trainedModel_r'+str(runId)+'*_py.mat')
        # to find what matModel includes use matModel.keys() --> trainedModel, trainPats, trainLabels
        # for each model we have W [ nVoxel x 2 classes], biases [ 1 x 2 classes]
        # we can't apply this model to any of the examples in this run, but let's apply it to the first 4 blocks of the next run
        # now load testing data from the next run to test it on
        pyModel_train = utils.loadMatFile(pyModelFn)
        # INSTEAD MAKE NEW MODEL
        lrc1 = LogisticRegression(penalty='l2', solver='saga',max_iter=300)
        lrc2 = LogisticRegression(penalty='l2', solver='saga',max_iter=300)

        lrc1.fit(pyModel_train.trainPats[:,ROI_indices], pyModel_train.trainLabels[:, 0])
        lrc2.fit(pyModel_train.trainPats[:,ROI_indices], pyModel_train.trainLabels[:, 1])
        newTrainedModel = utils.MatlabStructDict({}, 'trainedModel')
        newTrainedModel.trainedModel = StructDict({})
        newTrainedModel.trainedModel.weights = np.concatenate((lrc1.coef_.T, lrc2.coef_.T), axis=1)
        newTrainedModel.trainedModel.biases = np.concatenate((lrc1.intercept_, lrc2.intercept_)).reshape(1, 2)
        newTrainedModel.trainPats = pyModel_train.trainPats[:,ROI_indices]
        newTrainedModel.trainLabels = pyModel_train.trainLabels

        # now load testing data for CV
        pyModelFn = utils.findNewestFile(pyDataDir, 'trainedModel_r'+str(runId + 1)+'*_py.mat')
        pyModel_test = utils.loadMatFile(pyModelFn)
        py_test_data = pyModel_test.trainPats[nTRTest:,ROI_indices]
        test_labels = pyModel_test.trainLabels[nTRTest:,:]
        py_cs = np.zeros((nTRTest, 1))
        activations = np.zeros((nTRTest,2))
        for t in np.arange(nTRTest):
            _, _, _, activations_py = Test_L2_RLR_realtime(newTrainedModel,py_test_data[t,:],test_labels[t,:])
            activations[t,:] = activations_py

        fpr2, tpr2, thresholds2 = metrics.roc_curve(test_labels[:,1],activations[:,1] - activations[:,0],pos_label=1)
        auc_score[runId-1,d] = metrics.auc(fpr2,tpr2) # auc of this data applied to the first half of the next run
        # now apply to block data-- realtime values
        pyDataFn = utils.findNewestFile(pyDataDir, 'blkGroup_r' + str(runId + 1) + '_p2_*_py.mat')
        pyData_test = utils.loadMatFile(pyDataFn)
        regressor = pyData_test.regressor
        TRs_to_test = np.argwhere(np.sum(regressor,axis=0))
        RT_data = pyData_test.raw_sm_filt_z[TRs_to_test,ROI_indices]
        RT_regressor = regressor[:,TRs_to_test].T.reshape(nTRTest,2)
        # now do the same thing and test for every TR --> get category separation
        cs = np.zeros((nTRTest,))
        for t in np.arange(nTRTest):
            categ = np.flatnonzero(RT_regressor[t,:])
            otherCateg = (categ + 1) % 2
            _, _, _, activations_py = Test_L2_RLR_realtime(newTrainedModel,RT_data[t,:].flatten(),RT_regressor[t,:])
            cs[t] = activations_py[categ] - activations_py[otherCateg]
        RT_cs_timecourse[runId-1,:,d] = cs   
        # take average for this run
        RT_cs[runId-1, d] = np.mean(cs)

outfile = subjectDir + '/' 'offlineAUC_RTCS' + '_' + ROI    
np.savez(outfile,auc=auc_score,cs=RT_cs,csOverTime=RT_cs_timecourse)


