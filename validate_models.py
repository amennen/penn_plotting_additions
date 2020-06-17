import os
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

def validateMatlabPython(subjectNum,subjectDay):
	dataPath = '/data/jag/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo/'
	configFile = dataPath + 'subject' + str(subjectNum) + '/usedscripts/PennCfg_Day' + str(subjectDay) + '.toml'
	cfg = loadConfigFile(configFile)
	#subjectDayDir = getSubjectDayDir(cfg.session.subjectNum, cfg.session.subjectDay)
	subjectDayDir = '/data/jag/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo/subject' + str(cfg.session.subjectNum) + '/day' + str(cfg.session.subjectDay)
	matDataDir = subjectDayDir #os.path.join(cfg.session.dataDir, subjectDayDir)
	pyDataDir = matDataDir
	all_vals = np.zeros((200,2,len(cfg.session.Runs)))
	for runId in cfg.session.Runs:
		print("EXECUTING ANALYSES FOR RUN {}".format(runId))
		#validatePatternsData(matDataDir, pyDataDir, runId)
		mat_cs,py_cs = crossvalidateModels(matDataDir,pyDataDir,runId)
		# 200 TRs for each run --> want to plot
		all_vals[:,0,runId-1] = mat_cs
		all_vals[:,1,runId-1] = py_cs
	all_mat_ev = np.reshape(all_vals[:,0,:],(len(cfg.session.Runs)*200,1))
	all_py_ev = np.reshape(all_vals[:,1,:],(len(cfg.session.Runs)*200,1))
	fix,ax = plt.subplots(figsize=(12,7))
	plt.plot(all_mat_ev,all_py_ev, '.')
	plt.plot([-5,5],[-5,5], '--k')
	plt.title('S%i MAT x PY CORR = %4.4f' % (cfg.session.subjectNum, scipy.stats.pearsonr(all_mat_ev,all_py_ev)[0][0]))
	plt.xlabel('MATLAB')
	plt.ylabel('PYTHON')
	plt.xlim([-1.5,1.5])
	plt.ylim([-1.5,1.5])
	plt.show()
	#fullfilename = matDataDir + '/' + 'xvalresults.npy'
	#print("saving to %s\n" % fullfilename)
	#np.save(fullfilename,all_ROC)

def validateModelsMatlabPython(subjectNum,subjectDay,usesamedata):

	dataPath = '/data/jag/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo/'
	configFile = dataPath + 'subject' + str(subjectNum) + '/usedscripts/PennCfg_Day' + str(subjectDay) + '.toml'
	cfg = loadConfigFile(configFile)
	#subjectDayDir = getSubjectDayDir(cfg.session.subjectNum, cfg.session.subjectDay)
	subjectDayDir = '/data/jag/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo/subject' + str(cfg.session.subjectNum) + '/day' + str(cfg.session.subjectDay)
	matDataDir = subjectDayDir #os.path.join(cfg.session.dataDir, subjectDayDir)
	pyDataDir = matDataDir
	all_vals = np.zeros((100,2,cfg.session.Runs[-1]-1))
	usenewmodel = 1
	#usesamedata = 1 #whether or not to use same data as with matlab
	for runId in np.arange(1,cfg.session.Runs[-1]):
		runDir = 'run'+str(runId)+'/'
		matModelFn = utils.findNewestFile(matDataDir, runDir+'trainedModel_'+str(runId)+'*.mat')
		pyModelFn = utils.findNewestFile(pyDataDir, 'trainedModel_r'+str(runId)+'*_py.mat')
		matModel_train = utils.loadMatFile(matModelFn)
		# to find what matModel includes use matModel.keys() --> trainedModel, trainPats, trainLabels
		# for each model we have W [ nVoxel x 2 classes], biases [ 1 x 2 classes]
		# we can't apply this model to any of the examples in this run, but let's apply it to the first 4 blocks of the next run
		# now load testing data from the next run to test it on
		pyModel_train = utils.loadMatFile(pyModelFn)
		# INSTEAD MAKE NEW MODEL
		print(runId)
		if usenewmodel:
			lrc1 = LogisticRegression(penalty='l2', solver='sag',max_iter=300)
			lrc2 = LogisticRegression(penalty='l2', solver='sag',max_iter=300)
			if usesamedata:
				lrc1.fit(matModel_train.trainPats, pyModel_train.trainLabels[:, 0])
				lrc2.fit(matModel_train.trainPats, pyModel_train.trainLabels[:, 1])
			else:
				lrc1.fit(pyModel_train.trainPats, pyModel_train.trainLabels[:, 0])
				lrc2.fit(pyModel_train.trainPats, pyModel_train.trainLabels[:, 1])
			newTrainedModel = utils.MatlabStructDict({}, 'trainedModel')
			newTrainedModel.trainedModel = StructDict({})
			newTrainedModel.trainedModel.weights = np.concatenate((lrc1.coef_.T, lrc2.coef_.T), axis=1)
			newTrainedModel.trainedModel.biases = np.concatenate((lrc1.intercept_, lrc2.intercept_)).reshape(1, 2)
			newTrainedModel.trainPats = pyModel_train.trainPats
			newTrainedModel.trainLabels = pyModel_train.trainLabels
		# now load the models to test on
		matModelFn = utils.findNewestFile(matDataDir, 'run' + str(runId+1) + '/' + 'trainedModel_'+str(runId + 1)+'*.mat')
		pyModelFn = utils.findNewestFile(pyDataDir, 'trainedModel_r'+str(runId + 1)+'*_py.mat')
		matModel_test = utils.loadMatFile(matModelFn)
		pyModel_test = utils.loadMatFile(pyModelFn)
		nTRTest = 100
		mat_test_data = matModel_test.trainPats[nTRTest:,:]
		py_test_data = pyModel_test.trainPats[nTRTest:,:]
		test_labels = matModel_test.trainLabels[nTRTest:,:]
		mat_cs = np.zeros((nTRTest,1))
		py_cs = np.zeros((nTRTest,1))
		for t in np.arange(nTRTest):
			categ = np.flatnonzero(test_labels[t,:])
			otherCateg = (categ + 1) % 2 
			_, _, _, activations_mat = Test_L2_RLR_realtime(matModel_train,mat_test_data[t,:],test_labels[t,:])
			mat_cs[t] = activations_mat[categ] - activations_mat[otherCateg]
			if not usenewmodel:
				if not usesamedata:   	
					_, _, _, activations_py = Test_L2_RLR_realtime(pyModel_train,py_test_data[t,:],test_labels[t,:])
				else:
					_, _, _, activations_py = Test_L2_RLR_realtime(pyModel_train,mat_test_data[t,:],test_labels[t,:])
			else:
				if not usesamedata:
					_, _, _, activations_py = Test_L2_RLR_realtime(newTrainedModel,py_test_data[t,:],test_labels[t,:])
				else:
					_, _, _, activations_py = Test_L2_RLR_realtime(newTrainedModel,mat_test_data[t,:],test_labels[t,:])
			py_cs[t] = activations_py[categ] - activations_py[otherCateg]
		all_vals[:,0,runId-1] = mat_cs[:,0]
		all_vals[:,1,runId-1] = py_cs[:,0]
		#plt.figure()
		#if usenewmodel:
		#	plt.plot(matModel_train.weights[:,0],newTrainedModel.weights[:,0], '.')
		#else:
		#	plt.plot(matModel_train.weights[:,0],pyModel_train.weights[:,0], '.')
		#plt.xlim([-.02 ,.02])
		#plt.ylim([-.02 ,.02])
		#plt.xlabel('MATLAB')
		#plt.ylabel('PYTHON')
		#plt.show()
	all_mat_ev = np.reshape(all_vals[:,0,:],((cfg.session.Runs[-1]-1)*100,1))
	all_py_ev = np.reshape(all_vals[:,1,:],((cfg.session.Runs[-1]-1)*100,1))
	fix,ax = plt.subplots(figsize=(12,7))
	plt.plot(all_mat_ev,all_py_ev, '.')
	plt.plot([-5,5],[-5,5], '--k')
	plt.title('S%i MAT x PY CORR = %4.4f' % (cfg.session.subjectNum, scipy.stats.pearsonr(all_mat_ev,all_py_ev)[0][0]))
	plt.xlabel('MATLAB')
	plt.ylabel('PYTHON')
	plt.xlim([-1.5,1.5])
	plt.ylim([-1.5,1.5])
	plt.show()   
	
	plt.figure()
	plt.hist(all_mat_ev,alpha=0.6,label='matlab')
	plt.hist(all_py_ev, alpha=0.6,label='python')
	plt.xlabel('Correct - Incorrect Activation')
	plt.ylabel('Frequency')
	plt.title('S%i MAT x PY CORR = %4.4f' % (cfg.session.subjectNum, scipy.stats.pearsonr(all_mat_ev,all_py_ev)[0][0]))
	plt.legend()
	plt.show()
	


def crossvalidateModels(matDataDir, pyDataDir, runId):
    runDir = 'run'+str(runId)+'/'
    matModelFn = utils.findNewestFile(matDataDir, runDir+'trainedModel_'+str(runId)+'*.mat')
    pyModelFn = utils.findNewestFile(pyDataDir, 'trainedModel_r'+str(runId)+'*_py.mat')
    matModel = utils.loadMatFile(matModelFn)
    pyModel = utils.loadMatFile(pyModelFn)
    selector = np.concatenate((0*np.ones((50)),1*np.ones((50)),2*np.ones((50)),3*np.ones((50))),axis=0)
    X = np.array([1,2,3,4])
    nfold = 4
    kf = KFold(nfold)
    mat_cs = np.zeros((nfold,50))
    py_cs = np.zeros((nfold,50))
    i = 0
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        trTrain = np.in1d(selector,train_index)
        trTest = np.in1d(selector,test_index)
        # matlab first
        mat_lrc = LogisticRegression()
        categoryTrainLabels = np.argmax(matModel.trainLabels[trTrain,:],axis=1)
        mat_lrc.fit(matModel.trainPats[trTrain,:], categoryTrainLabels)
        mat_predict = mat_lrc.predict_proba(matModel.trainPats[trTest,:])
        categ_sep = -1*np.diff(mat_predict,axis=1)
        C0 = np.argwhere(np.argmax(matModel.trainLabels[trTest,:],axis=1)==0)
        C1 = np.argwhere(np.argmax(matModel.trainLabels[trTest,:],axis=1)==1)
        C1_label = C1.flatten()
        mat_correct_subtraction = categ_sep.flatten()
        mat_correct_subtraction[C1_label] = -1*mat_correct_subtraction[C1_label]
        # python second
        py_lrc = LogisticRegression()
        categoryTrainLabels = np.argmax(pyModel.trainLabels[trTrain,:],axis=1)
        py_lrc.fit(pyModel.trainPats[trTrain,:], categoryTrainLabels)
        py_predict = py_lrc.predict_proba(pyModel.trainPats[trTest,:])
        categ_sep = -1*np.diff(py_predict,axis=1)
        C0 = np.argwhere(np.argmax(pyModel.trainLabels[trTest,:],axis=1)==0)
        C1 = np.argwhere(np.argmax(pyModel.trainLabels[trTest,:],axis=1)==1)
        C1_label = C1.flatten()
        py_correct_subtraction = categ_sep.flatten()
        py_correct_subtraction[C1_label] = -1*py_correct_subtraction[C1_label]
        mat_cs[i,:] = mat_correct_subtraction
        py_cs[i,:] = py_correct_subtraction
        
        i+= 1
    mat_corr = mat_cs.flatten()
    py_corr = py_cs.flatten()
    return mat_corr,py_corr
    
def validateRTMatlabPython(subjectNum,subjectDay):
	d1_runs = 6
	if subjectNum == 106:
		d1_runs = 5
	d2_runs = 8
	d3_runs = 7
	totalRuns = d1_runs + d2_runs + d3_runs
	nsubjects = len(subjects)
	all_mat_data = np.zeros((nsubjects,totalRuns*100)) # TO DO: MAKE THIS AND DO LINEAR PLOT
	rtAttenPath = '/data/jag/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'

