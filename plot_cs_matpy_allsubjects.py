# plot results/look at group differences

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
import scipy
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
# for each subject, you need to run getcs.py in anne_additions first to get cs evidence for that subject
# have python and matlab versions--let's start with matlab 
# WANT: plot for EACH SUBJECT ALL RUNS BY DAY MATLAB/PYTHON MEAN-- one fig/subject
#toUse = 'mat' # whether to use matlab or python in coding

#subjects = np.array([1,2,101, 102,103,104,105,106])
subjects = np.array([3,106]) # want to check for new subjects
HC_subjects=subjects[subjects<=100]
n_HC = len(HC_subjects)
MDD_subjects=subjects[subjects>100]
n_MDD = len(MDD_subjects)
d1_runs = 6
d2_runs = 8
d3_runs = 7
nsubjects = len(subjects)
#all_mat_data = np.zeros((nsubjects,totalRuns*100)) # TO DO: MAKE THIS AND DO LINEAR PLOT
rtAttenPath = '/data/jag/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'

# get HC averages for each RUN OF SCANNER/DAY
# cs matrices are n runs x 100 TRs x 3 days
for s in np.arange(len(subjects)):
	subjectNum = subjects[s]
	subjectDir = rtAttenPath + '/' + 'subject' + str(subjectNum)
	outfile = subjectDir + '/' 'realtimeevidence.npz'    
	z=np.load(outfile)
	cs_mat = z['mat']
	cs_py = z['py']
	if subjectNum == 106:
		d1_runs = 5
	else:
		d1_runs = 6
	totalRuns = d1_runs + d2_runs + d3_runs 
	mat_run_avg_day1 = np.mean(cs_mat[0:d1_runs,:,0],axis=1)
	py_run_avg_day1 = np.mean(cs_py[0:d1_runs,:,0],axis=1)
	mat_run_avg_day2 = np.mean(cs_mat[0:d2_runs,:,0],axis=1)
	py_run_avg_day2 = np.mean(cs_py[0:d2_runs,:,0],axis=1)
	mat_run_avg_day3 = np.mean(cs_mat[0:d3_runs,:,0],axis=1)
	py_run_avg_day3 = np.mean(cs_py[0:d3_runs,:,0],axis=1)
	mat_run_average = np.concatenate((mat_run_avg_day1,mat_run_avg_day2,mat_run_avg_day3))
	py_run_average = np.concatenate((py_run_avg_day1,py_run_avg_day2,py_run_avg_day3))
	#fix,ax = plt.subplots(figsize=(12,7))
	#ind = np.arange(len(mat_run_average))
	#alpha=0.5
	#plt.bar(ind,mat_run_average,alpha=alpha,label='MATLAB', color='r')
	#plt.bar(ind,py_run_average,alpha=alpha,label='PYTHON', color='g')
	#plt.title('All run evidence for subject %i' % subjectNum)
	#plt.xlabel('Run number')
	#plt.ylabel('Classification Evidence')
	#plt.legend()
	#plt.show()
	# instead: plot scatter
	days_mat = np.concatenate((cs_mat[0:d1_runs,:,0],cs_mat[0:d2_runs,:,1],cs_mat[0:d3_runs,:,2]),axis=0)
	days_py = np.concatenate((cs_py[0:d1_runs,:,0],cs_py[0:d2_runs,:,1],cs_py[0:d3_runs,:,2]),axis=0)
	all_mat_ev = np.reshape(days_mat,(totalRuns*100,1))
	all_py_ev = np.reshape(days_py,(totalRuns*100,1))
	fix,ax = plt.subplots(figsize=(12,7))
	plt.plot(all_mat_ev,all_py_ev, '.')
	plt.plot([-5,5],[-5,5], '--k')
	plt.title('S%i MAT x PY CORR = %4.4f' % (subjectNum, scipy.stats.pearsonr(all_mat_ev,all_py_ev)[0][0]))
	plt.xlabel('MATLAB')
	plt.ylabel('PYTHON')
	plt.xlim([-1.5,1.5])
	plt.ylim([-1.5,1.5])
	plt.show()
	



# now just the beginnign of the first and last of the last
# for s in np.arange(len(subjects)):
# 	subjectDir = rtAttenPath + '/' + 'subject' + str(subjects[s])
# 	outfile = subjectDir + '/' 'realtimeevidence.npz'    
# 	z=np.load(outfile)
# 	cs = z[toUse]
# 	if subjects[s] in HC_subjects:
# 		line1=plt.plot(np.array([1,3]), np.array([np.mean(cs[0:2,:,0]),np.mean(cs[d3_runs-2:d3_runs,:,2])]), '-')
# 	if subjects[s] in MDD_subjects:
# 		line2=plt.plot(np.array([1,3]), np.array([np.mean(cs[0:2,:,0]),np.mean(cs[d3_runs-2:d3_runs,:,2])]), ':')
# hc_avg = plt.plot(np.array([1,3]),np.mean(HC_tp_average,axis=0),'k-')
# mdd_avg = plt.plot(np.array([1,3]),np.mean(MDD_tp_average,axis=0),'k:')
# plt.title('Average, first 2 and last 2 runs')
# plt.show()
