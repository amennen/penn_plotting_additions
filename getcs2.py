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

def getCategSepData(subjectNum):

        rtAttenPath = '/data/jag/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'

        subjectDir = rtAttenPath + '/' + 'subject' + str(subjectNum)
        # this will be less annoying but will just have to average and remember padding zeros 
        all_matlab_evidence = np.zeros((9,100,3))
        all_python_evidence = np.zeros((9,100,3))
        for d in np.arange(3):
                subjectDay = d + 1
                matDataDir= rtAttenPath + '/' + 'subject' + str(subjectNum) + '/' + 'day' + str(subjectDay)
                pyDataDir= rtAttenPath + '/' + 'subject' + str(subjectNum) + '/' + 'day' + str(subjectDay)

                if subjectDay==1:
                        # then we have 7 runs
                        nRuns=7
                        if subjectNum == 106:
                                nRuns=6
                elif subjectDay==2:
                        nRuns=9
                elif subjectDay==3:
                        nRuns=8
                #n_feedback_runs = nRuns - 1 # no feedback in first run
                print(np.arange(2,nRuns+1))

                        #def getCategSepData(matDataDir, pyDataDir, runId):
                for RUN in np.arange(2,nRuns+1):
                        runId = RUN # because 0 index, skip first run
                        runDir = 'run'+str(runId)+'/'
                        # Check how well raw_sm_filt_z values match
                        matPatternsFn = utils.findNewestFile(matDataDir, runDir+'patternsdata_'+str(runId)+'*.mat')
                        pyBlkGrp1Fn = utils.findNewestFile(pyDataDir, 'blkGroup_r'+str(runId)+'_p1_*_py.mat')
                        pyBlkGrp2Fn = utils.findNewestFile(pyDataDir, 'blkGroup_r'+str(runId)+'_p2_*_py.mat')
                        print("Getting rt classifer data from: Matlab {}, Python {} {}".format(matPatternsFn, pyBlkGrp1Fn, pyBlkGrp2Fn))

                        matPatterns = utils.loadMatFile(matPatternsFn)
                        pyBlkGrp1 = utils.loadMatFile(pyBlkGrp1Fn)
                        pyBlkGrp2 = utils.loadMatFile(pyBlkGrp2Fn)
                        mat_nTRs = matPatterns.raw.shape[0]
                        pyp1_nTRs = pyBlkGrp1.raw.shape[0]
                        pyp2_nTRs = pyBlkGrp2.raw.shape[0]
                        py_nTRs = pyp1_nTRs + pyp2_nTRs
                        mat_nVoxels = matPatterns.raw.shape[1]
                        py_nVoxels = pyBlkGrp1.raw.shape[1]

                        if mat_nTRs != py_nTRs or mat_nVoxels != py_nVoxels:
                                raise ValidationError("Number of TRs or Voxels don't match: nTRs m{} p{}, nVoxels m{} p{}".
                                                                          format(mat_nTRs, py_nTRs, mat_nVoxels, py_nVoxels))

                        matPatterns.categoryseparation
                        relevant_TR = np.argwhere(np.sum(matPatterns.regressor,0))[:,0]
                        RT_TR = relevant_TR[int(len(relevant_TR)/2):]
                        mat_RT_CS = matPatterns.categoryseparation[:,RT_TR][0,:]

                        pyCombined_categoryseparation = np.full((py_nTRs,), np.nan)
                        pyCombined_categoryseparation[0:pyp1_nTRs] = pyBlkGrp1.categoryseparation
                        pyCombined_categoryseparation[pyp1_nTRs:] = pyBlkGrp2.categoryseparation
                        py_RT_CS = pyCombined_categoryseparation[RT_TR]

                        all_matlab_evidence[RUN-2,:,d] = mat_RT_CS
                        all_python_evidence[RUN-2,:,d] = py_RT_CS
                        # now you have 2 (100,) arrays with the category separation from that feedback run

        outfile = subjectDir + '/' 'realtimeevidence'
        np.savez(outfile,mat=all_matlab_evidence,py=all_python_evidence)



def train_test_python_classifier(subjectNum):
    ndays = 3
    auc_score = np.zeros((8,ndays)) # save larger to fit all days in
    RT_cs = np.zeros((8,ndays))
    dataPath = '/data/jag/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo/'
    subjectDir =  dataPath + '/' + 'subject' + str(subjectNum)
    print(subjectNum)
    all_python_evidence = np.zeros((9,100,3)) # time course of classifier evidence
    for d in np.arange(ndays):
        print(d)
        subjectDay = d + 1
        configFile = dataPath + 'subject' + str(subjectNum) + '/usedscripts/PennCfg_Day' + str(subjectDay) + '.toml'
        cfg = loadConfigFile(configFile)
        subjectDayDir = '/data/jag/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo/subject' + str(cfg.session.subjectNum) + '/day' + str(cfg.session.subjectDay)
        pyDataDir = subjectDayDir
        if subjectDay == 1:
            nRuns = 7
            print('here')
            if str(subjectNum) == '106':
                nRuns = 6
                print('here')
            else:
                print(subjectNum)
                if subjectNum == 106:
                    print('finding it here')
                print('nothere')
        elif subjectDay == 2:
            nRuns = 9
        elif subjectDay == 3:
            nRuns = 8
        print('total number of runs: %i' % nRuns)
        print(subjectNum)
        print(subjectDay)
        print(nRuns)
        #nruns = len(cfg.session.Runs) - 1
        #nruns = len(cfg.session.Runs) - 1
        for r in np.arange(0,nRuns-1):
            runId = r + 1 # now it goes from 0 : n Runs - 1
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

            lrc1.fit(pyModel_train.trainPats, pyModel_train.trainLabels[:, 0])
            lrc2.fit(pyModel_train.trainPats, pyModel_train.trainLabels[:, 1])
            newTrainedModel = utils.MatlabStructDict({}, 'trainedModel')
            newTrainedModel.trainedModel = StructDict({})
            newTrainedModel.trainedModel.weights = np.concatenate((lrc1.coef_.T, lrc2.coef_.T), axis=1)
            newTrainedModel.trainedModel.biases = np.concatenate((lrc1.intercept_, lrc2.intercept_)).reshape(1, 2)
            newTrainedModel.trainPats = pyModel_train.trainPats
            newTrainedModel.trainLabels = pyModel_train.trainLabels

            # now load testing data for CV
            pyModelFn = utils.findNewestFile(pyDataDir, 'trainedModel_r'+str(runId + 1)+'*_py.mat')
            pyModel_test = utils.loadMatFile(pyModelFn)
            nTRTest = 100
            py_test_data = pyModel_test.trainPats[nTRTest:,:]
            test_labels = pyModel_test.trainLabels[nTRTest:,:]
            py_cs = np.zeros((nTRTest, 1))
            activations = np.zeros((nTRTest,2))
            for t in np.arange(nTRTest):
                    _, _, _, activations_py = Test_L2_RLR_realtime(newTrainedModel,py_test_data[t,:],test_labels[t,:])
                    activations[t,:] = activations_py

            fpr2, tpr2, thresholds2 = metrics.roc_curve(test_labels[:,1],activations[:,1] - activations[:,0],pos_label=1)
            auc_score[r,d] = metrics.auc(fpr2,tpr2) # auc of this data applied to the first half of the next run
            # now apply to block data-- realtime values
            pyDataFn = utils.findNewestFile(pyDataDir, 'blkGroup_r' + str(runId + 1) + '_p2_*_py.mat')
            pyData_test = utils.loadMatFile(pyDataFn)
            regressor = pyData_test.regressor
            TRs_to_test = np.argwhere(np.sum(regressor,axis=0))
            RT_data = pyData_test.raw_sm_filt_z[TRs_to_test,:]
            RT_regressor = regressor[:,TRs_to_test].T.reshape(nTRTest,2)
            # now do the same thing and test for every TR --> get category separation
            cs = np.zeros((nTRTest,1))
            for t in np.arange(nTRTest):
                    categ = np.flatnonzero(RT_regressor[t,:])
                    otherCateg = (categ + 1) % 2
                    _, _, _, activations_py = Test_L2_RLR_realtime(newTrainedModel,RT_data[t,:].flatten(),RT_regressor[t,:])
                    cs[t] = activations_py[categ] - activations_py[otherCateg]

            # take average for this run
            RT_cs[r, d] = np.mean(cs)
            all_python_evidence[r,:,d] = cs[:,0]
    outfile = subjectDir + '/' 'offlineAUC_RTCS'
    np.savez(outfile,auc=auc_score,cs=RT_cs,all_ev=all_python_evidence)

