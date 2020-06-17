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
import scipy
font = {'weight' : 'normal',
        'size'   : 22}
import pandas as pd
import seaborn as sns
matplotlib.rc('font', **font)
# for each subject, you need to run getcs.py in anne_additions first to get cs evidence for that subject
# have python and matlab versions--let's start with matlab 

from anne_additions.plotting_pretty.commonPlotting import *

subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101, 102,103,104,105,106, 107,108,109,110,111,112,113])
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
first_run_score = np.zeros((nsubs,3))
last_run_score = np.zeros((nsubs,3))
rtAttenPath = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'

# get HC averages for each RUN OF SCANNER/DAY
nDays = 3
for s in np.arange(nsubs):
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
            neg_pts = np.where(categSep[run,:]<-.1)[0] # if consecutive negative
            pos_pts = np.where(categSep[run,:]>.1)[0]

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
        first_run_score[s,d] = run_avg_neg[0]
        last_run_score[s,d] = run_avg_neg[run]
        day_avg_neg[d] = np.mean(run_avg_neg)
        day_avg_pos[d] = np.mean(run_avg_pos)
    day_average_neg[s,:] = day_avg_neg
    day_average_pos[s,:] = day_avg_pos

stat = day_average_neg
fig = plotPosterStyle(stat,subjects)
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.01)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(stat),0.01)
x,y = nonNan(stat[HC_ind,2],stat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,2,np.nanmax(stat),0.01)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat(p/2,0,2,np.nanmax(stat),2)
plt.ylabel('# negative in a row')
plt.xticks(np.arange(3))
plt.show()


firstrun = first_run_score.flatten()
lastrun = last_run_score.flatten()
day = np.tile(np.arange(nDays),nsubs)
subject = np.repeat(subjects,nDays)
groups = ['HC' if i in HC_ind else 'MDD' for i in np.arange(nsubs)]
groups = np.repeat(groups,nDays)
DATA = {}
DATA['firstrun'] = firstrun
DATA['lastrun'] = lastrun
DATA['day'] = day
DATA['subject'] = subject
DATA['groups'] = groups
df = pd.DataFrame.from_dict(DATA)

# different version for poster--divide by day
pal = dict(HC='k', MDD='r')
g = sns.FacetGrid(df,col='day',palette=pal)
g.map(plt.scatter,'groups','firstrun',color=['k'],alpha=0.3)
g.map(sns.pointplot,'groups','firstrun',palette=pal,ci=68,alpha=0.5,scale=1.5,errwidth=5)
plt.ylim([0, 25])
g.add_legend()
plt.show()

scipy.stats.ttest_ind(first_run_score[MDD_ind,0],first_run_score[HC_ind,0])
scipy.stats.ttest_ind(first_run_score[MDD_ind,2],first_run_score[HC_ind,2])

pal = dict(HC='k', MDD='r')
g = sns.FacetGrid(df,col='day',palette=pal)
g.map(plt.scatter,'groups','lastrun',color=['k'],alpha=0.3)
g.map(sns.pointplot,'groups','lastrun',palette=pal,ci=68,alpha=0.5,scale=1.5,errwidth=5)
g.add_legend()
plt.show()

linestyles = ['-', ':']
colors=['k', 'r']
nVisits = 3



#fig = plt.figure(figsize=(10,20))

plt.subplot(121)
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
plt.ylim([0,20])
plt.title('Mean negative points by day')
plt.legend()
#plt.show()

colors=['k', 'r']
nVisits = 3

#fig = plt.figure(figsize=(10,7))
plt.subplot(122)
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),day_average_pos[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(day_average_pos[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(day_average_pos[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(day_average_pos[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(day_average_pos[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('NF 1', 'NF 2', 'NF 3'))
plt.xlabel('Visit')
plt.ylabel('# Pos in a row')
plt.ylim([0,20])
plt.title('Mean positive points by day')
plt.legend()
plt.show()


# now plot for each subject neg run x AUC average

plt.figure()
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(day_average[s,:],day_average_neg[s,:], marker='.',ms=20,color=colors[style],alpha=0.5,lw=0)
plt.errorbar(np.nanmean(day_average[HC_ind,:],axis=0),np.nanmean(day_average_neg[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(day_average_neg[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.nanmean(day_average[MDD_ind,:],axis=0),np.nanmean(day_average_neg[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(day_average_neg[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
#plt.xticks(np.arange(nVisits),('NF 1', 'NF 2', 'NF 3'))
plt.xlabel('AUC Day Average')
plt.ylabel('# Neg in a row')
#plt.ylim([0,20])
plt.title('Mean negative points by day')
plt.legend()
plt.show()

# subject average

plt.figure()
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.mean(day_average[s,:]),np.mean(day_average_neg[s,:]), marker='.',ms=20,color=colors[style],alpha=0.5,lw=0)
#plt.errorbar(np.nanmean(day_average[HC_ind,:],axis=0),np.nanmean(day_average_neg[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(day_average_neg[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
#plt.errorbar(np.nanmean(day_average[MDD_ind,:],axis=0),np.nanmean(day_average_neg[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(day_average_neg[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
#plt.xticks(np.arange(nVisits),('NF 1', 'NF 2', 'NF 3'))
plt.xlabel('AUC Average over all days')
plt.ylabel('Mean negative points over all days')
#plt.ylim([0,20])
plt.title('Mean negative points by day')
plt.legend()
plt.show()

########################## these next two plots are comparing average classifier evidence with AUC
plt.figure()
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.mean(day_average[s,:]),np.mean(cs_day_average[s,:]), marker='.',ms=20,color=colors[style],alpha=0.5,lw=0)
#plt.errorbar(np.nanmean(day_average[HC_ind,:],axis=0),np.nanmean(day_average_neg[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(day_average_neg[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
#plt.errorbar(np.nanmean(day_average[MDD_ind,:],axis=0),np.nanmean(day_average_neg[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(day_average_neg[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
#plt.xticks(np.arange(nVisits),('NF 1', 'NF 2', 'NF 3'))
plt.xlabel('AUC average over all days')
plt.ylabel('Mean CS during NF over all days')
#plt.ylim([0,20])
plt.title('Mean negative points by day')
plt.legend()
plt.show()

plt.figure()
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(day_average[s,:],cs_day_average[s,:], marker='.',ms=20,color=colors[style],alpha=0.5,lw=0)
#plt.errorbar(np.nanmean(day_average[HC_ind,:],axis=0),np.nanmean(day_average_neg[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(day_average_neg[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
#plt.errorbar(np.nanmean(day_average[MDD_ind,:],axis=0),np.nanmean(day_average_neg[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(day_average_neg[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
#plt.xticks(np.arange(nVisits),('NF 1', 'NF 2', 'NF 3'))
plt.xlabel('Mean AUC in day')
plt.ylabel('Mean CS by day')
#plt.ylim([0,20])
plt.title('Mean negative points by day')
plt.legend()
plt.show()
