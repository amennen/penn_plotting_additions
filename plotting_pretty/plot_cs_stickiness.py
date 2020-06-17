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
# AVGERAGE NUMBER OF CONSECUTIVE NEGATIVE TIME POINTS 
# AVERAGE NUMBER OF CONSECUTIVE POSITIVE TIME POINTS
toUse = 'mat' # whether to use matlab or python in coding

subjects = np.array([1,2,3,4,101, 102,103,104,105, 106,107,108])
HC_subjects=subjects[subjects<=100]
n_HC = len(HC_subjects)
MDD_subjects=subjects[subjects>100]
n_MDD = len(MDD_subjects)
d1_runs = 6
d2_runs = 8
d3_runs = 7
nDays=3
totalRuns = d1_runs + d2_runs + d3_runs
HC_day_average = np.zeros((n_HC,3))
MDD_day_average = np.zeros((n_MDD,3))
HC_day_average_pos = np.zeros((n_HC,3))
MDD_day_average_pos = np.zeros((n_MDD,3))

rtAttenPath = '/data/jag/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'

# get HC averages for each RUN OF SCANNER/DAY

for s in np.arange(n_HC):
	subjectDir = rtAttenPath + '/' + 'subject' + str(HC_subjects[s])
	outfile = subjectDir + '/' 'realtimeevidence.npz'    
	z=np.load(outfile)
	cs = z[toUse]
	nTR = np.shape(cs)[1]
	day_avg_neg = np.zeros((nDays,))
	day_avg_pos = np.zeros((nDays,))

	for d in np.arange(nDays):
		time_cp = {}
		if d == 0:
			categSep = cs[0:d1_runs,:,0]
			nRuns = d1_runs
		elif d == 1:
			categSep = cs[0:d2_runs,:,1]
			nRuns = d2_runs
		elif d == 2:
			categSep = cs[0:d3_runs,:,2]
			nRuns = d3_runs
		run_avg_neg = np.zeros((nRuns,))
		run_avg_pos = np.zeros((nRuns,))

		for run in np.arange(nRuns):
			
			neg_pts = np.where(categSep[run,:]<0)[0] # if consecutive negative
			pos_pts = np.where(categSep[run,:]>0)[0]
			
			time_cp[run] = []
			accounted_for_pts = []
			for n in np.arange(len(neg_pts)):
				if neg_pts[n] not in accounted_for_pts:
					# find positive point closest to neg_pts
					this_neg = neg_pts[n]
					pos_neg_diff = np.where(pos_pts-this_neg > 0)[0]
					if pos_neg_diff.size > 0:
						closestIndex = np.where(pos_pts - this_neg > 0)[0][0]
						closestPoint = pos_pts[closestIndex]
						time_elapsed = closestPoint - this_neg
						time_cp[run].append(time_elapsed)
						# put any points in between into the accounted for category
						accounted_for_pts = np.arange(closestPoint+1)
					else: # there are no  more positive points--negative string goes to the end
						time_elapsed = np.max(neg_pts) - neg_pts[n] + 1
						time_cp[run].append(time_elapsed)
						accounted_for_pts = np.arange(np.max(neg_pts)+1)
			run_avg_neg[run] = np.mean(time_cp[run])
			
			time_cp[run] = []
			accounted_for_pts = []
			for n in np.arange(len(pos_pts)):
				if pos_pts[n] not in accounted_for_pts:
					# find positive point closest to neg_pts
					this_pos = pos_pts[n]
					pos_neg_diff = np.where(neg_pts-this_pos > 0)[0]
					if pos_neg_diff.size > 0:
						closestIndex = np.where(neg_pts - this_pos > 0)[0][0]
						closestPoint = neg_pts[closestIndex]
						time_elapsed = closestPoint - this_pos
						time_cp[run].append(time_elapsed)
						# put any points in between into the accounted for category
						accounted_for_pts = np.arange(closestPoint+1)
					else: # there are no  more positive points--negative string goes to the end
						time_elapsed = np.max(pos_pts) - pos_pts[n] + 1
						time_cp[run].append(time_elapsed)
						accounted_for_pts = np.arange(np.max(pos_pts)+1)
			run_avg_pos[run] = np.mean(time_cp[run])				
		day_avg_neg[d] = np.mean(run_avg_neg)
		day_avg_pos[d] = np.mean(run_avg_pos)
	HC_day_average[s,:] = day_avg_neg
	HC_day_average_pos[s,:] = day_avg_pos

for s in np.arange(n_MDD):
	subjectDir = rtAttenPath + '/' + 'subject' + str(MDD_subjects[s])
	outfile = subjectDir + '/' 'realtimeevidence.npz'    
	z=np.load(outfile)
	cs = z[toUse]
	nTR = np.shape(cs)[1]
	day_avg = np.zeros((nDays,))
	for d in np.arange(nDays):
		time_cp = {}
		if d == 0:
			categSep = cs[0:d1_runs,:,0]
			nRuns = d1_runs
			if MDD_subjects[s] == 106:
				categSep = cs[0:d1_runs-1,:,0]
				nRuns = 5
		elif d == 1:
			categSep = cs[0:d2_runs,:,1]
			nRuns = d2_runs
		elif d == 2:
			categSep = cs[0:d3_runs,:,2]
			nRuns = d3_runs
		run_avg_neg = np.zeros((nRuns,))
		run_avg_pos = np.zeros((nRuns,))		
		for run in np.arange(nRuns):
			neg_pts = np.where(categSep[run,:]<0)[0] # if consecutive negative
			pos_pts = np.where(categSep[run,:]>0)[0]
			
			time_cp[run] = []
			accounted_for_pts = []
			for n in np.arange(len(neg_pts)):
				if neg_pts[n] not in accounted_for_pts:
					# find positive point closest to neg_pts
					this_neg = neg_pts[n]
					pos_neg_diff = np.where(pos_pts-this_neg > 0)[0]
					if pos_neg_diff.size > 0:
						closestIndex = np.where(pos_pts - this_neg > 0)[0][0]
						closestPoint = pos_pts[closestIndex]
						time_elapsed = closestPoint - this_neg
						time_cp[run].append(time_elapsed)
						# put any points in between into the accounted for category
						accounted_for_pts = np.arange(closestPoint+1)
					else: # there are no  more positive points--negative string goes to the end
						time_elapsed = np.max(neg_pts) - neg_pts[n] + 1
						time_cp[run].append(time_elapsed)
						accounted_for_pts = np.arange(np.max(neg_pts)+1)
			run_avg_neg[run] = np.mean(time_cp[run])
			
			time_cp[run] = []
			accounted_for_pts = []
			for n in np.arange(len(pos_pts)):
				if pos_pts[n] not in accounted_for_pts:
					# find positive point closest to neg_pts
					this_pos = pos_pts[n]
					pos_neg_diff = np.where(neg_pts-this_pos > 0)[0]
					if pos_neg_diff.size > 0:
						closestIndex = np.where(neg_pts - this_pos > 0)[0][0]
						closestPoint = neg_pts[closestIndex]
						time_elapsed = closestPoint - this_pos
						time_cp[run].append(time_elapsed)
						# put any points in between into the accounted for category
						accounted_for_pts = np.arange(closestPoint+1)
					else: # there are no  more positive points--negative string goes to the end
						time_elapsed = np.max(pos_pts) - pos_pts[n] + 1
						time_cp[run].append(time_elapsed)
						accounted_for_pts = np.arange(np.max(pos_pts)+1)
			run_avg_pos[run] = np.mean(time_cp[run])				
		day_avg_neg[d] = np.mean(run_avg_neg)
		day_avg_pos[d] = np.mean(run_avg_pos)
	
	MDD_day_average[s,:] = day_avg_neg
	MDD_day_average_pos[s,:] = day_avg_pos

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
plt.title('NEGATIVE Scene evidence during RTNF by day')
plt.ylabel('Scene - Face evidence')
ax.set_xticks(ind)
ax.set_xticklabels(['Day 1', 'Day 2', 'Day 3'])
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12,7))
alpha=0.5
lw=2
ind = np.arange(1,4)
for s in np.arange(n_HC):
	line1=plt.plot(ind,HC_day_average_pos[s,:], '-',linewidth=lw)
for s in np.arange(n_MDD):
	line2=plt.plot(ind,MDD_day_average_pos[s,:], ':',linewidth=lw)
#hc_avg = plt.plot(ind,np.mean(HC_day_average,axis=0),'k-')
#mdd_avg = plt.plot(ind,np.mean(MDD_day_average,axis=0),'k:')
plt.bar(ind,np.mean(HC_day_average_pos,axis=0),alpha=alpha,label='HC', color='k')
plt.bar(ind,np.mean(MDD_day_average_pos,axis=0),alpha=alpha,label='MDD', color='r')
plt.title('POSITIVE Scene evidence during RTNF by day')
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
