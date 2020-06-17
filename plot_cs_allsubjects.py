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
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
# for each subject, you need to run getcs.py in anne_additions first to get cs evidence for that subject
# have python and matlab versions--let's start with matlab 

toUse = 'py' # whether to use matlab or python in coding

subjects = np.array([1,2,3,4,101, 102,103,104,105, 106,107])
HC_subjects=subjects[subjects<=100]
n_HC = len(HC_subjects)
MDD_subjects=subjects[subjects>100]
n_MDD = len(MDD_subjects)
d1_runs = 6
d2_runs = 8
d3_runs = 7
totalRuns = d1_runs + d2_runs + d3_runs
HC_run_average = np.zeros((n_HC,totalRuns))
MDD_run_average = np.zeros((n_MDD,totalRuns))
HC_day_average = np.zeros((n_HC,3))
MDD_day_average = np.zeros((n_MDD,3))
HC_tp_average = np.zeros((n_HC,2))
MDD_tp_average = np.zeros((n_MDD,2))

rtAttenPath = '/data/jag/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'

# get HC averages for each RUN OF SCANNER/DAY

for s in np.arange(n_HC):
	subjectDir = rtAttenPath + '/' + 'subject' + str(HC_subjects[s])
	outfile = subjectDir + '/' 'realtimeevidence.npz'    
	z=np.load(outfile)
	cs = z[toUse]
	day1avg = np.mean(cs[0:d1_runs,:,0],axis=1)
	day2avg = np.mean(cs[0:d2_runs,:,1],axis=1)
	day3avg = np.mean(cs[0:d3_runs,:,2],axis=1)
	
	HC_day_average[s,:] = np.array([np.mean(day1avg),np.mean(day2avg),np.mean(day3avg)])
	HC_run_average[s,:] = np.concatenate((day1avg,day2avg,day3avg))
	HC_tp_average[s,0] = np.mean(cs[0:2,:,0])
	HC_tp_average[s,1] = np.mean(cs[d3_runs-2:d3_runs,:,2])

for s in np.arange(n_MDD):
	subjectDir = rtAttenPath + '/' + 'subject' + str(MDD_subjects[s])
	outfile = subjectDir + '/' 'realtimeevidence.npz'    
	z=np.load(outfile)
	cs = z[toUse]
	day1avg = np.mean(cs[0:d1_runs,:,0],axis=1)
	day2avg = np.mean(cs[0:d2_runs,:,1],axis=1)
	day3avg = np.mean(cs[0:d3_runs,:,2],axis=1)
	MDD_day_average[s,:] = np.array([np.mean(day1avg),np.mean(day2avg),np.mean(day3avg)])
	MDD_run_average[s,:] = np.concatenate((day1avg,day2avg,day3avg))
	MDD_tp_average[s,0] = np.mean(cs[0:2,:,0])
	MDD_tp_average[s,1] = np.mean(cs[d3_runs-2:d3_runs,:,2])	


# now do same thing and average by day
fig, ax = plt.subplots(figsize=(12,7))
alpha=0.5
lw=2
ind = np.arange(1,4)
for s in np.arange(n_HC):
	line1=plt.plot(ind,HC_day_average[s,:], '-',linewidth=lw)
for s in np.arange(n_MDD):
	line2=plt.plot(ind,MDD_day_average[s,:], ':',linewidth=lw)
#hc_avg = plt.plot(ind,np.mean(HC_day_average,axis=0),'k-')
#mdd_avg = plt.plot(ind,np.mean(MDD_day_average,axis=0),'k:')
plt.bar(ind,np.mean(HC_day_average,axis=0),alpha=alpha,label='HC', color='k')
plt.bar(ind,np.mean(MDD_day_average,axis=0),alpha=alpha,label='MDD', color='r')
plt.title('Scene evidence during RTNF by day')
plt.ylabel('Scene - Face evidence')
ax.set_xticks(ind)
ax.set_xticklabels(['Day 1', 'Day 2', 'Day 3'])
plt.legend()
plt.show()

# now create plot by runs

# plt.figure()
# for s in np.arange(n_HC):
# 	plt.plot(HC_run_average[s,:], '--')
# for s in np.arange(n_MDD):
# 	plt.plot(MDD_run_average[s,:])
# hc_avg = plt.plot(np.mean(HC_run_average,axis=0),'k--')
# mdd_avg = plt.plot(np.mean(MDD_run_average,axis=0),'k')
# plt.legend((hc_avg,mdd_avg),('HC', 'MDD'))
# plt.show()





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
