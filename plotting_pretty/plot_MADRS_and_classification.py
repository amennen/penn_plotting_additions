# plot MADRS changes and neurofeedback
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
import scipy
import csv
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 22,
        'sans-serif': 'Geneva'}

matplotlib.rc('font', **font)
# for each subject, you need to run getcs.py in anne_additions first to get cs evidence for that subject
# have python and matlab versions--let's start with matlab 


subjects = np.array([1,2,101,102,103,105,105,106,3,107,4,108,5,6,109,7,110,8,9,10,11,111,112])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
nsubs = len(subjects)
d1_runs = 6
d2_runs = 8
d3_runs = 7
totalRuns = d1_runs + d2_runs + d3_runs

# now do the same thing for cs
day_average = np.zeros((nsubs,3))
day_average_neg = np.zeros((nsubs,3))
day_average_pos = np.zeros((nsubs,3))
cs_day_average = np.zeros((nsubs,3))

rtAttenPath = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'

# get HC averages for each RUN OF SCANNER/DAY
nDays = 3
for s in np.arange(nsubs):
#for s in np.arange(1):
#	s=0
	subjectDir = rtAttenPath + '/' + 'subject' + str(subjects[s])
	outfile = subjectDir + '/' 'offlineAUC_RTCS.npz'    
	z=np.load(outfile)
	AUC = z['auc']

	if subjects[s] == 106:
		d1_runs = 5
	else:
		d1_runs = 6
	day1avg = np.mean(AUC[0:d1_runs,0])
	day2avg = np.mean(AUC[0:d2_runs,1])
	day3avg = np.mean(AUC[0:d3_runs,2])
	day_average[s,:] = np.array([day1avg,day2avg,day3avg])
	CS = z['cs']
	day1avg = np.mean(CS[0:d1_runs,0])
	day2avg = np.mean(CS[0:d2_runs,1])
	day3avg = np.mean(CS[0:d3_runs,2])
	cs_day_average[s,:] = np.array([day1avg,day2avg,day3avg])
	CS = z['csOverTime']
	nTR = np.shape(CS)[1]
	day_avg_neg = np.zeros((nDays,))
	day_avg_pos = np.zeros((nDays,))

	for d in np.arange(nDays):
		time_cp = {}
		if d == 0:
			categSep = CS[0:d1_runs,:,0]
			nRuns = d1_runs
		elif d == 1:
			categSep = CS[0:d2_runs,:,1]
			nRuns = d2_runs
		elif d == 2:
			categSep = CS[0:d3_runs,:,2]
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
	day_average_neg[s,:] = day_avg_neg
	day_average_pos[s,:] = day_avg_pos



projectDir = '/data/jux/cnds/amennen/rtAttenPenn/'
csv_fn = projectDir + 'MADRS.csv'
nVisits = 4
MADRS_SCORES = {}
with open(csv_fn) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if len(row) > 0:
                if 'RT' in row[0]:
                        subject = row[0]
                        goodrow=row
                        subject_scores = np.zeros((nVisits,))
                        subject_scores.fill(np.nan)
                        nInfo = len(goodrow)
                        for v in np.arange(2,nInfo):
                                if len(goodrow[v]) > 0:
                                        subject_scores[v-2] = np.int(goodrow[v])
                        MADRS_SCORES[subject] = subject_scores

snames = [*MADRS_SCORES]
scores_differences = np.zeros((nsubs,3))
for s in np.arange(nsubs):
    snumber = subjects[s]
    sname = snames[s]
    subject_scores = MADRS_SCORES[sname]
    scores_differences[s,0] = subject_scores[1] - subject_scores[0]
    scores_differences[s,1] = subject_scores[2] - subject_scores[0]
    scores_differences[s,2] = subject_scores[3] - subject_scores[0]

# AUC vs. MADRS change
day_average_diff = day_average[:,2] - day_average[:,0]
day_average_neg_diff = day_average_neg[:,2] - day_average_neg[:,0]
colors=['k', 'r']
plt.figure()
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(day_average_diff[s],-1*scores_differences[s,0], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.xlabel('Improvement in AUC')
plt.ylabel('Improvement in depression severity')
plt.show()

print(scipy.stats.pearsonr(day_average_diff[MDD_ind],-1*scores_differences[MDD_ind,0]))

colors=['k', 'r']
#plt.figure(figsize=(15,11))
plt.figure(figsize=(12,7))
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        if s == 0:
            plt.plot(-1*day_average_neg_diff[s],-1*scores_differences[s,0], marker='.',ms=20,color=colors[style],alpha=0.5,label='healthy')
        elif s == 2:
            plt.plot(-1*day_average_neg_diff[s],-1*scores_differences[s,0], marker='.',ms=20,color=colors[style],alpha=0.5,label='depressed')
        plt.plot(-1*day_average_neg_diff[s],-1*scores_differences[s,0], marker='.',ms=20,color=colors[style],alpha=0.5)
y = -1*scores_differences[:,0]
x=-1*day_average_neg_diff[~np.isnan(y)]
y=y[~np.isnan(y)]
#plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='k', alpha=0.6, lw=10)
plt.xlabel('Improvement in negative attention during neurofeedback')
plt.ylabel('Improvement in depression severity')
plt.ylim([-15,15])
plt.legend(loc=2)
plt.show()
print(scipy.stats.pearsonr(day_average_neg_diff[MDD_ind],scores_differences[MDD_ind,0]))

colors=['k', 'r']
#plt.figure(figsize=(15,11))
plt.figure(figsize=(12,7))
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        if s == 0:
            plt.plot(day_average_diff[s],-1*day_average_neg_diff[s] ,marker='.',ms=20,color=colors[style],alpha=0.5,label='healthy')
        elif s == 2:
            plt.plot(day_average_diff[s],-1*day_average_neg_diff[s], marker='.',ms=20,color=colors[style],alpha=0.5,label='depressed')
        plt.plot(day_average_diff[s],-1*day_average_neg_diff[s], marker='.',ms=20,color=colors[style],alpha=0.5)

#plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='k', alpha=0.6, lw=10)
plt.xlabel('Improvement in negative attention during neurofeedback')
plt.xlabel('Improvement in AUC')
#plt.ylim([-15,15])
plt.legend(loc=2)
plt.show()
print(scipy.stats.pearsonr(-1*day_average_neg_diff[MDD_ind],day_average_diff[MDD_ind]))



linestyles = ['-', ':']
colors=['k', 'r']
nVisits = 3

plt.figure(figsize=(15,11))
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),day_average_neg[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(day_average_neg[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(day_average_neg[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(day_average_neg[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(day_average_neg[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('NF 1', 'NF 2', 'NF 3'))
plt.xlabel('Visit')
plt.ylabel('# Neg in a row')
plt.ylim([0,15])
plt.title('Mean negative points by day')
plt.legend()
plt.show()

print(scipy.stats.ttest_ind(day_average_neg[HC_ind,0],day_average_neg[MDD_ind,0]))
