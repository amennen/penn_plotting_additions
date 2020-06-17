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
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
# for each subject, you need to run getcs.py in anne_additions first to get cs evidence for that subject
# have python and matlab versions--let's start with matlab 


subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101, 102,103,104,105,106, 107,108,109,110,111,112])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
nsubs = len(subjects)
n_HC = len(HC_ind)
n_MDD = len(MDD_ind)
d1_runs = 6
d2_runs = 8
d3_runs = 7
totalRuns = d1_runs + d2_runs + d3_runs
#HC_run_average = np.zeros((n_HC,totalRuns))
#MDD_run_average = np.zeros((n_MDD,totalRuns))
day_average = np.zeros((nsubs,3))
# now do the same thing for cs
cs_day_average = np.zeros((nsubs,3))

#HC_tp_average = np.zeros((n_HC,2))
#MDD_tp_average = np.zeros((n_MDD,2))

rtAttenPath = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'

# get HC averages for each RUN OF SCANNER/DAY

for s in np.arange(nsubs):
#for s in np.arange(1):
#   s=0
    subjectDir = rtAttenPath + '/' + 'subject' + str(subjects[s])
    outfile = subjectDir + '/' 'offlineAUC_RTCS.npz'    
    z=np.load(outfile)
    AUC = z['auc']
    if subjects[s] == 106:
        d1_runs = 5
    else:
        d1_runs = 6

    print(s)
    print('day 1')
    print(AUC[0:d1_runs,0])
    print('day 2')
    print(AUC[0:d2_runs,1]) 
    print('day 3')
    print(AUC[0:d3_runs,2])
    day1avg = np.mean(AUC[0:d1_runs,0])
    day2avg = np.mean(AUC[0:d2_runs,1])
    day3avg = np.mean(AUC[0:d3_runs,2])
    
    day_average[s,:] = np.array([day1avg,day2avg,day3avg])

    CS = z['cs']
    print('day 1')
    print(CS[0:d1_runs,0])
    print('day 2')
    print(CS[0:d2_runs,1])  
    print('day 3')
    print(CS[0:d3_runs,2])
    day1avg = np.mean(CS[0:d1_runs,0])
    day2avg = np.mean(CS[0:d2_runs,1])
    day3avg = np.mean(CS[0:d3_runs,2])
    cs_day_average[s,:] = np.array([day1avg,day2avg,day3avg])
    #HC_run_average[s,:] = np.concatenate((day1avg,day2avg,day3avg))

linestyles = ['-', ':']
colors=['k', 'r']
nVisits = 3

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



fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),day_average[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(day_average[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(day_average[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(day_average[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(day_average[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('NF 1', 'NF 2', 'NF 3'))
plt.xlabel('Visit')
plt.ylabel('AUC')
plt.title('Stable blocks AUC')
plt.legend()
plt.show()


fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),cs_day_average[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(cs_day_average[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(cs_day_average[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(cs_day_average[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(cs_day_average[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('NF 1', 'NF 2', 'NF 3'))
plt.xlabel('Visit')
plt.ylabel('Scene - Face Evidence')
plt.title('Scene - Face Evidence during NF')
plt.legend()
plt.show()


# now do the same thing for category separation





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
#   plt.plot(HC_run_average[s,:], '--')
# for s in np.arange(n_MDD):
#   plt.plot(MDD_run_average[s,:])
# hc_avg = plt.plot(np.mean(HC_run_average,axis=0),'k--')
# mdd_avg = plt.plot(np.mean(MDD_run_average,axis=0),'k')
# plt.legend((hc_avg,mdd_avg),('HC', 'MDD'))
# plt.show()





# now just the beginnign of the first and last of the last
# for s in np.arange(len(subjects)):
#   subjectDir = rtAttenPath + '/' + 'subject' + str(subjects[s])
#   outfile = subjectDir + '/' 'realtimeevidence.npz'    
#   z=np.load(outfile)
#   cs = z[toUse]
#   if subjects[s] in HC_subjects:
#       line1=plt.plot(np.array([1,3]), np.array([np.mean(cs[0:2,:,0]),np.mean(cs[d3_runs-2:d3_runs,:,2])]), '-')
#   if subjects[s] in MDD_subjects:
#       line2=plt.plot(np.array([1,3]), np.array([np.mean(cs[0:2,:,0]),np.mean(cs[d3_runs-2:d3_runs,:,2])]), ':')
# hc_avg = plt.plot(np.array([1,3]),np.mean(HC_tp_average,axis=0),'k-')
# mdd_avg = plt.plot(np.array([1,3]),np.mean(MDD_tp_average,axis=0),'k:')
# plt.title('Average, first 2 and last 2 runs')
# plt.show()

