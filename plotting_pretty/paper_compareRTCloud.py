# Purpose: compare final NF --> cloud based offline scores

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
#rom sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from rtfMRI.StructDict import StructDict, MatlabStructDict
from sklearn.metrics import roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
import scipy
font = {'weight' : 'normal',
        'size'   : 22}
plt.rc('axes', linewidth=2)

import csv
from anne_additions.plotting_pretty.commonPlotting import *
from anne_additions.aprime_file im

def getCSFromBlockData(subjectNum,day,run,block):
  """day, run, block are in MATLAB 1-based indices, but block is the run starting with 1 as first real-time run"""
  rtAttenPath = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'
  subjectDir = rtAttenPath + '/' + 'subject' + str(subjectNum)
  dayDir = subjectDir + '/' + 'day' + str(day)
  runDir = dayDir + '/' + 'run' + str(run)
  file = glob.glob(os.path.join(runDir,'blockdata_*.mat'))[0]
  d = utils.loadMatFile(file)
  all_categsep = d.blockData.categsep
  block_categsep = all_categsep[:,block-1 +4][0][0]
  block_found = d.blockData.classOutputFile[:,block-1+4][0][0]
  n_failed = 0
  test = True
  for b in np.arange(len(block_found)):
  	this_str = block_found[b][0]
  	if 'notload' in this_str:
  		  n_failed += 1
  if n_failed > 0:
  	test = False
  # when_read = np.argwhere(~np.isnan(block_categsep))[:,0]
  # fileRead = d.blockData.newestFile[:,block-1+4][0][0]
  # # for f in np.arange(len(when_read)):
  # # 	print(fileRead[when_read[f]])
  # fInd = np.arange(2,49,2)
  # fInd = np.arange(8,49,2)
  # first_file = fileRead[fInd]
  # all_nums = np.zeros((len(fInd),))*np.nan
  # for z in np.arange(len(fInd)):
  # 	this_num = first_file[z][0]
  # 	n = int(this_num[4:7])
  # 	all_nums[z] = n
  # all_diff = np.diff(all_nums)
  # test = True
  # if np.any(all_diff == 0):
  # 	test = False
  # if np.any(all_diff > 1):
  # 	test = False
  # first_file = first_file[0].split('.')[0]
  # str_file = first_file[0:7]
  # all_first = np.array([126, 155, 183, 211])
  # expected_first = 'vol_{0}'.format(all_first[block-1])
  # if first_file <= expected_first:
  # 	test = True
  # elif first_file > expected_first:
  # 	test = False
  x,_ = nonNan(block_categsep,[])
  return x,test#,expected_first,first_file,fileRead

def getCSFromFile(subjectNum):
  rtAttenPath = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'
  subjectDir = rtAttenPath + '/' + 'subject' + str(subjectNum)
  outfile = subjectDir + '/' 'offlineAUC_RTCS.npz' 
  z=np.load(outfile)
  CS = z['csOverTime']
  return CS

def getCSBlockData(CS,run,day,block):
  # run is 1-based as used in Matlab
  runInd = run - 1 - 1 #  to go from Matlab run --> python NF index
  # block is 1-based
  blockInd = block - 1
  # day is 1-based
  dayInd = day - 1
  # runInd is 0-based
  nTRpBlock = 25
  CS_blockData = CS[runInd,blockInd*nTRpBlock:(blockInd+1)*nTRpBlock,dayInd]
  return CS_blockData


subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,12,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115])
n_days = 3
d1_runs = 6
d2_runs = 8
d3_runs = 7
blockNums = np.array([1,2,3,4])
subject_corr = np.zeros((len(subjects),d2_runs,len(blockNums),n_days))*np.nan
n_bad = 0
n_good = 0
for s in np.arange(len(subjects)):
	this_subject = subjects[s]
	CS_P_all = getCSFromFile(this_subject)

	for d in np.arange(ndays):
		this_day = d + 1 # 1-based
		if this_day == 1:
			if this_subject == 106:
				nf_runs = 5
			else:
				nf_runs = d1_runs
		elif this_day == 2:
			nf_runs = d2_runs
		elif this_day == 3:
			nf_runs = d3_runs
		for r in np.arange(nf_runs):
			this_run = r + 1 + 1 # because we're going to matlab indexing, for the actual run number
			block_corr = np.zeros((4,))*np.nan
			for b in np.arange(len(blockNums)):
				this_block = blockNums[b]	
				# run and block in matlab terms, but block is 1 starting as the first real-time run
				CS_M,testRes = getCSFromBlockData(this_subject,this_day,this_run,this_block)
				n_m = len(CS_M)
				CS_P = getCSBlockData(CS_P_all,this_run,this_day,this_block)[0:len(CS_M)]
				block_corr = scipy.stats.pearsonr(CS_M,CS_P)[0]
				#if s == 26 and r == 0 and b == 1 and d == 2:
				#	# try shifting over 1 if it was late

				if block_corr < 0.85:
					print('***********************')
					print(block_corr)
					print(s,r,b,d)
					print(testRes)
					n_bad += 1
				else:
					n_good += 1
					if not testRes:
						print('!!!!!!!!!!!!!!!!!!!')
						print(s,r,b,d)
						print('good but failed test!')

				subject_corr[s,r,b,d] = block_corr

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
# how many blocks per day:
subject_corr_vec,_ = nonNan(subject_corr.flatten(),[])
mean_confidence_interval(subject_corr_vec)
(d1_runs+d2_runs+d3_runs)*len(subjects)*4 -4

np.nanmin(subject_corr)
np.nanmax(subject_corr)
np.nanmean(subject_corr)

this_subject=115
this_run = 2 # actual run that day
this_block = 4
CS_M = getCSFromBlockData(this_subject,this_day,this_run,this_block)
CS_P_all = getCSFromFile(this_subject)
CS_P = getCSBlockData(CS_P_all,this_run,this_day,this_block)[0:len(CS_M)]


