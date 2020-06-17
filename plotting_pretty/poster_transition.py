 #plot results/look at group differences

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
import csv
from anne_additions.plotting_pretty.commonPlotting import *
matplotlib.rc('font', **font)
# for each subject, you need to run getcs.py in anne_additions first to get cs evidence for that subject
# have python and matlab versions--let's start with matlab 
def getMADRSscoresALL():
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
                          subjectID = row[0]
                          subjectNumber = np.int(row[1])
                          goodrow=row
                          subject_scores = np.zeros((nVisits,))
                          subject_scores.fill(np.nan)
                          nInfo = len(goodrow)
                          for v in np.arange(2,nInfo):
                                  if len(goodrow[v]) > 0:
                                          subject_scores[v-2] = np.int(goodrow[v])
                          MADRS_SCORES[subjectNumber] = subject_scores
  return MADRS_SCORES

def getMADRSdiff(MADRS_SCORES,allsubjects):
  nSubs = len(allsubjects)
  diff_v5_v1 = np.zeros((nSubs,))
  diff_v6_v1 = np.zeros((nSubs,))
  diff_v7_v1 = np.zeros((nSubs,))

  for s in np.arange(nSubs):
    subjectNum  = allsubjects[s]
    this_sub_madrs = MADRS_SCORES[subjectNum]
    diff_v5_v1[s] = this_sub_madrs[1] - this_sub_madrs[0]
    diff_v6_v1[s] = this_sub_madrs[2] - this_sub_madrs[0]
    diff_v7_v1[s] = this_sub_madrs[3] - this_sub_madrs[0]
  return diff_v5_v1,diff_v6_v1,diff_v7_v1


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def transition_matrix(transitions,nstates):
    #n = 1+ max(transitions) #number of states
    n=nstates
    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101, 102,103,104,105,106, 107,108,109,110,111,112,113,114])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
nsubs = len(subjects)
d1_runs = 6
d2_runs = 8
d3_runs = 7
totalRuns = d1_runs + d2_runs + d3_runs


all_x = {}
all_dx = {}
rtAttenPath = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'
master_x = []
# get HC averages for each RUN OF SCANNER/DAY
nDays = 3
takeHalf = 1
for s in np.arange(nsubs):
    subject_key = 'subject' + str(subjects[s])
    all_x[subject_key] = {}
    all_dx[subject_key] = {}
    subjectDir = rtAttenPath + '/' + 'subject' + str(subjects[s])
    outfile = subjectDir + '/' 'offlineAUC_RTCS.npz'    
    z=np.load(outfile)
    if subjects[s] == 106:
        d1_runs = 5
    else:
        d1_runs = 6
    CS = z['csOverTime']
    nTR = np.shape(CS)[1]
    for d in np.arange(nDays):
        day_x = []
        day_dx = []
        if d == 0:
            if takeHalf:
                d1_runs = 3
            categSep = CS[0:d1_runs,:,0]
            nRuns = np.shape(categSep)[0]
        elif d == 1:
            categSep = CS[0:d2_runs,:,1]
            nRuns = np.shape(categSep)[0]
        elif d == 2:
            if takeHalf:
                categSep = CS[4:d3_runs,:,2]
            else:
                categSep = CS[0:d3_runs,:,2]
            nRuns = np.shape(categSep)[0]
        vec = categSep.flatten()
        day_x.append(vec)
        vec_diff = np.diff(vec)
        #a = np.array([0])
        #day_dx.append(np.concatenate([a,vec_diff]))
        day_dx.append(vec_diff)
        day_key = 'day' + str(d)
        all_x[subject_key][day_key] = day_x
        master_x.extend(day_x[0])
# first make giant histogram
np.arange(-1,-0.75,0.025)
np.arange(0.8,1.05, 0.025)
bins = [-1.   , -0.975, -0.9, -0.8 ,-0.7,-0.55,-0.4,-0.2,0,0.2,0.4,0.55,0.7, 0.8 ,  0.9 , 0.975, 1. ]
plt.figure()
n = plt.hist(master_x,bins)
labels_pos = np.array(bins).astype(np.float)
labels_pos = np.around(labels_pos,decimals=2).astype(np.str)
plt.xticks(bins,labels_pos,fontsize=8)
plt.ylabel('counts in range')
plt.show()

# now separate
nbins=len(bins)

# after meeting with Ken on 11/7 --> change distribution to be uniform
# first pass: use everyones 
step_size=0.2
#pos_edges = np.arange(-1,1+step_size,step_size)
#nbins = len(pos_edges)
all_matrices = np.zeros((nbins-1,nbins-1,nsubs,nDays))
for s in np.arange(nsubs):
    subject_key = 'subject' + str(subjects[s])
    for d in np.arange(nDays):
        day_key = 'day' + str(d)
        pos = all_x[subject_key][day_key][0]
        #indices = np.digitize(pos,pos_edges)
        indices = np.digitize(pos,bins)
        # if evidence is exactly 1.0 will be outside range, so scale back to 10
        #indices[np.argwhere(indices==len(pos_edges))] = len(pos_edges) - 1
        indices[np.argwhere(indices==len(bins))] = len(bins) - 1
        indices_0ind = indices.copy() - 1
        # now make sure from 0 - 9
        # want to sort into bin[x] --> see what 
        M = np.array(transition_matrix(indices_0ind,nbins-1))
        all_matrices[:,:,s,d] = M # [1:,1:] - now made index 0 based so don't need to do this
        if len(np.unique(indices_0ind)) != nbins-1:
            print('subject %i, day %i' % (s,d))
            print('len n states = %i' % len(np.unique(indices_0ind)))
            values_taken = np.unique(indices_0ind)
            other = [x for x in np.arange(nbins-1) if x not in values_taken]
            nbad = len(other)
            for ibad in np.arange(nbad):
                all_matrices[other[ibad],:,s,d] = np.nan
        # check that the matrix probabilities sum to 1
        this_sum = np.nansum(all_matrices[:,:,s,d],axis=1)
        err=1E-10
        if len(np.argwhere(np.abs(this_sum-1) > err)):
            print('subject %i, day %i' % (s,d))
            print('BAD - NOT SUMMING TO 1')
        # check histogram for each subject
        #plt.figure()
        #plt.hist(pos,bins)
        #plt.show()
        #for row in M: print(' '.join('{0:.2f}'.format(x) for x in row))
        #for row in all_matrices[:,:,s,d]: print(' '.join('{0:.2f}'.format(x) for x in row))
# plot average transition matrix by day
#labels_pos = pos_edges.astype(np.float)
#labels_pos = np.around(labels_pos,decimals=3).astype(np.str)
labels_pos = np.array(bins).astype(np.float)
labels_pos = np.around(labels_pos,decimals=2).astype(np.str)
vmin=0
vmax=1
#day 1 first plt.figure(figsize=(20,20))
d=0
# make sequential colormap
fig,ax = plt.subplots(figsize=(20,20))
plt.subplot(1,3,1)
this_plot_hc = np.nanmean(all_matrices[:,:,HC_ind,d],axis=2)
plt.imshow(this_plot_hc,cmap='Reds',vmin=vmin,vmax=vmax)
#plt.colorbar()
plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('HC',fontsize=20)
plt.subplot(1,3,2)
this_plot_mdd = np.nanmean(all_matrices[:,:,MDD_ind,d],axis=2)
plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.imshow(this_plot_mdd,cmap='Reds',vmin=vmin,vmax=vmax)
plt.xlabel('value B')
plt.title('MDD',fontsize=20)
#plt.colorbar()
plt.show()
# to do: understand output of transition matrix -- which is A/which is B, plot for each day
plt.figure(figsize=(10,10))
plt.imshow(this_plot_mdd-this_plot_hc,cmap='bwr',vmin=-.1,vmax=.1) # for half, max diff is .2, for all days, max diff is 0.1
plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('MDD - HC')
plt.colorbar()
plt.show()

# now do the same by day by group
this_plot_hc1 = np.nanmean(all_matrices[:,:,HC_ind,0],axis=2)
this_plot_hc3 = np.nanmean(all_matrices[:,:,HC_ind,2],axis=2)
this_plot_mdd1 = np.nanmean(all_matrices[:,:,MDD_ind,0],axis=2)
this_plot_mdd3 = np.nanmean(all_matrices[:,:,MDD_ind,2],axis=2)
plt.figure(figsize=(10,10))
plt.imshow(this_plot_hc3-this_plot_hc1,cmap='bwr',vmin=-.16,vmax=.16) # for half, max diff is .16, for all days, max diff is 0.1
plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('HC2 - HC0')
plt.colorbar()
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(this_plot_mdd3-this_plot_mdd1,cmap='bwr',vmin=-.11,vmax=.11) # for half, max diff is .11, for all days, max diff is 0.06
plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('MDD2 - MDD0')
plt.colorbar()
plt.show()



# try to understand how distirbutions are changing--> maybe average shift by group?
# first plot one person's

ind_bin = 0
#plt.figure()
fig,ax = plt.subplots()
for s in np.arange(nsubs):
    if s in HC_ind:
        plt.subplot(1,2,1)
        matrix1 = all_matrices[ind_bin,:,s,0]
        matrix3 = all_matrices[ind_bin,:,s,2]
        plt.plot(np.arange(nbins-1),matrix1,color='c',alpha=0.5)
        plt.plot(np.arange(nbins-1),matrix3,color='b',alpha=0.5)
        plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
        plt.xlabel('value B')
        plt.ylabel('p(B|-1)')
        #plt.legend()
    elif s in MDD_ind:
        plt.subplot(1,2,2)
        matrix1 = all_matrices[ind_bin,:,s,0]
        matrix3 = all_matrices[ind_bin,:,s,2]
        plt.plot(np.arange(nbins-1),matrix1,color='c',alpha=0.5)
        plt.plot(np.arange(nbins-1),matrix3,color='b',alpha=0.5)
        plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
        plt.xlabel('value B')
        plt.ylabel('p(B|-1)')
plt.subplot(1,2,1)
hc_m1_avg = np.mean(all_matrices[ind_bin,:,HC_ind,0],axis=0)
hc_m3_avg = np.mean(all_matrices[ind_bin,:,HC_ind,2],axis=0)
plt.plot(np.arange(nbins-1),hc_m1_avg,color='c',alpha=1,lw=5,label='day 1')
plt.plot(np.arange(nbins-1),hc_m3_avg,color='b',alpha=1, lw=5,label='day 3')
plt.title('HC Group')
plt.legend()
plt.subplot(1,2,2)
mdd_m1_avg = np.mean(all_matrices[ind_bin,:,MDD_ind,0],axis=0)
mdd_m3_avg = np.mean(all_matrices[ind_bin,:,MDD_ind,2],axis=0)
plt.plot(np.arange(nbins-1),mdd_m1_avg,color='c',alpha=0.8,lw=5,label='day 1')
plt.plot(np.arange(nbins-1),mdd_m3_avg,color='b',alpha=0.8, lw=5,label='day 3')
plt.title('MDD Group')
plt.legend()
plt.show()

# TO DO NEXT: DO THE SAMET THING FOR POSITIVE ONES 

# plot individual difference in 9 x 9 matrix
d=2
labels = []
plt.figure()
ind=0
for i in np.arange(3):
    for j in np.arange(3):
        for s in np.arange(nsubs):
            this_sub_val = all_matrices[i,j,s,d]
            if s in HC_ind:
                color = 'k'
            elif s in MDD_ind:
                color = 'r'
            plt.plot(np.random.normal(ind, 0.1, 1)[0],this_sub_val, '.', ms=10,color=color,alpha=0.5)
        # plot means too
        plt.plot(ind,np.mean(all_matrices[i,j,HC_ind,d]),'*', ms=15, color='k')
        plt.plot(ind,np.mean(all_matrices[i,j,MDD_ind,d]),'*', ms=15, color='r')
        print('row=%i' % i)
        print('col=%i' % j)
        label = '%2.2f -> %2.2f' % (bins[i],bins[j])
        print('%2.2f -> %2.2f' % (bins[i],bins[j]))
        labels.append(label)
        ind+=1
plt.xticks(np.arange(ind),labels,fontsize=8)
plt.ylabel('p(transition)')
plt.show()
scipy.stats.ttest_ind(all_matrices[0,0,HC_ind,d],all_matrices[0,0,MDD_ind,d])

# now do positive bottom right 9 x 9 matrix
d=0
labels = []
plt.figure()
ind=0
for i in np.arange(3):
    for j in np.arange(3):
        for s in np.arange(nsubs):
            this_sub_val = all_matrices[i+13,j+13,s,d]
            if s in HC_ind:
                color = 'k'
            elif s in MDD_ind:
                color = 'r'
            plt.plot(np.random.normal(ind, 0.1, 1)[0],this_sub_val, '.', ms=10,color=color,alpha=0.5)
        # plot means too
        plt.plot(ind,np.mean(all_matrices[i+13,j+13,HC_ind,d]),'*', ms=15, color='k')
        plt.plot(ind,np.mean(all_matrices[i+13,j+13,MDD_ind,d]),'*', ms=15, color='r')
        print('row=%i' % (i+13))
        print('col=%i' % (j+13))
        label = '%2.2f -> %2.2f' % (bins[i+13],bins[j+13])
        print('%2.2f -> %2.2f' % (bins[i+13],bins[j+13]))
        labels.append(label)
        ind+=1
plt.xticks(np.arange(ind),labels,fontsize=8)
plt.ylabel('p(transition)')
plt.show()
scipy.stats.ttest_ind(all_matrices[0,0,HC_ind,d],all_matrices[0,0,MDD_ind,d])

# get average of upper square
upper_left = np.mean(np.mean(all_matrices[0:2,0:2,:,:],axis=0),axis=0)
lower_right = np.mean(np.mean(all_matrices[13:15,13:15,:,:],axis=0),axis=0)

data_mat = lower_right
fig = plotPosterStyle(data_mat,subjects)
plt.ylim([0,1])
x,y = nonNan(data_mat[MDD_ind,0],data_mat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat(p/2,0,2,np.nanmax(data_mat),0.05)
x,y = nonNan(data_mat[HC_ind,0],data_mat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(data_mat),0.01)
x,y = nonNan(data_mat[HC_ind,2],data_mat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,2,np.nanmax(data_mat),0.01)
plt.show()


# plot diff of upper square both groups over 1 --> 3
i=0
j=0
# i,j 0.2 is for -1-->-0.9
# try to stay positive = 15 15
data_mat = all_matrices[i,j,:,:]

fig = plotPosterStyle(data_mat,subjects)
plt.ylim([0,1])
x,y = nonNan(data_mat[MDD_ind,0],data_mat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat(p/2,0,2,np.nanmax(data_mat),0.05)
x,y = nonNan(data_mat[HC_ind,0],data_mat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(data_mat),0.01)
x,y = nonNan(data_mat[HC_ind,2],data_mat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,2,np.nanmax(data_mat),0.01)
plt.show()

# plot difference of MADRS vs. others
M = getMADRSscoresALL()
d1,d2,d3 = getMADRSdiff(M,subjects)
all_neg_change = data_mat[:,2] - data_mat[:,0]

colors = ['k', 'r'] # HC, MDD
fig = plt.figure(figsize=(10,7))
for s in np.arange(nsubs):
  subjectNum  = subjects[s]
  madrs_change = d1[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(-1*all_neg_change[s],-1*d1[s],marker='.',ms=20,color=colors[style],alpha=0.5)
plt.xlabel('Improvement in negative stickiness 3 - 1')
plt.ylabel('Improvement in MADRS: V5 -> V1')
#plt.xlim([-0.4,0.4])
plt.show()
# relationship in general becomes stronger with looking at first 4 runs, last 4 runs
# then positive relationship is stronger with increase in positive scene activation instead of looking at negative decrease
# staying positive improvement tracks MADRS improvement (BUT larger difference in constant negative)
x,y = nonNan(-1*d1[MDD_ind],all_neg_change[MDD_ind])
scipy.stats.pearsonr(x,y)

x,y = nonNan(-1*d1,all_neg_change)
scipy.stats.pearsonr(x,y)

scipy.stats.pearsonr(-1*d1[MDD_ind],all_neg_change[MDD_ind])
scipy.stats.pearsonr(d1[MDD_ind2],data_mat[MDD_ind2,0])
scipy.stats.pearsonr(-1*d1,all_neg_change)


np.random.normal(np.arange(2), 0.1, 2)




# NOW PLOT MDD 3 - MDD 1
mdd_1 = np.nanmean(all_matrices[:,:,MDD_ind,0],axis=2)
mdd_3 = np.nanmean(all_matrices[:,:,MDD_ind,2],axis=2)
plt.figure(figsize=(10,10))
plt.imshow(mdd_3-mdd_1,cmap='bwr',vmin=-0.08,vmax=0.08)
plt.yticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.xticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('MDD_3 - MDD_1')
plt.colorbar()
plt.show()

hc_1 = np.nanmean(all_matrices[:,:,HC_ind,0],axis=2)
hc_3 = np.nanmean(all_matrices[:,:,HC_ind,2],axis=2)
plt.figure(figsize=(10,10))
plt.imshow(hc_3-hc_1,cmap='bwr',vmin=-0.08,vmax=0.08)
plt.yticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.xticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('HC_3 - HC_1')
plt.colorbar()
plt.show()
# newest - get most likely denisty for each subject
# position will span -1 to 1 and velocity can span -2 to 2
most_likely_counts = {}
step_size=0.2
pos_edges = np.arange(-1,1+step_size,step_size)
vel_edges = np.arange(-2,2+step_size*2,step_size*2)
all_hist = np.zeros((len(pos_edges)-1,len(vel_edges)-1,nsubs,nDays))
for s in np.arange(nsubs):
    subject_key = 'subject' + str(subjects[s])
    most_likely_counts[subject_key] = {}
    for d in np.arange(nDays):
        day_key = 'day' + str(d)
        pos = all_x[subject_key][day_key][0]
        vel = all_dx[subject_key][day_key][0]
        H, xedges, yedges = np.histogram2d(pos, vel, bins=(pos_edges, vel_edges),density=False)
        H_norm = H/len(pos)
        all_hist[:,:,s,d] = H_norm
        #plt.imshow(H)
        #plt.show()
        most_counts = np.unravel_index(np.argmax(H),np.shape(H))
        most_likely_counts[subject_key][day_key] = most_counts

# now plot each
# day 1
fig,ax = plt.subplots()
plt.subplot(3,2,1)
this_plot = np.nanmean(all_hist[:,:,HC_ind,0],axis=2)
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.imshow(this_plot,cmap='Blues')
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.subplot(3,2,2)
this_plot = np.nanmean(all_hist[:,:,MDD_ind,0],axis=2)
plt.imshow(this_plot,cmap='Blues')
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.subplot(3,2,3)
this_plot = np.nanmean(all_hist[:,:,HC_ind,1],axis=2)
plt.imshow(this_plot,cmap='Blues')
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.subplot(3,2,4)
this_plot = np.nanmean(all_hist[:,:,MDD_ind,1],axis=2)
plt.imshow(this_plot,cmap='Blues')
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.subplot(3,2,5)
this_plot = np.nanmean(all_hist[:,:,HC_ind,2],axis=2)
plt.imshow(this_plot,cmap='Blues')
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.subplot(3,2,6)
this_plot = np.nanmean(all_hist[:,:,MDD_ind,2],axis=2)
plt.imshow(this_plot,cmap='Blues')
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.colorbar()

plt.show()
labels_vel = vel_edges.astype(np.float)
labels_vel= np.around(labels_vel,decimals=3).astype(np.str)
labels_pos = pos_edges.astype(np.float)
labels_pos = np.around(labels_pos,decimals=3).astype(np.str)
vmin=-0.015
vmax=0.015
#fig,ax = plt.subplots()
#plt.subplot(1,3,1)
plt.figure(figsize=(20,20))
d=0
this_plot_hc = np.nanmean(all_hist[:,:,HC_ind,d],axis=2).T
this_plot_mdd = np.nanmean(all_hist[:,:,MDD_ind,d],axis=2).T
plt.imshow(this_plot_mdd-this_plot_hc,cmap='bwr',vmin=vmin,vmax=vmax)
plt.yticks(np.arange(len(vel_edges))-0.5,labels_vel,fontsize=10)
plt.xticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.ylabel('d(CS)/dt')
plt.xlabel('CS: scene - face evidence',fontsize=15)
plt.title('Day %i' % d)
#plt.axis('equal')
plt.colorbar()

plt.show()
#plt.subplot(1,3,2)
plt.figure(figsize=(20,10))
d=1
this_plot_hc = np.nanmean(all_hist[:,:,HC_ind,d],axis=2).T
this_plot_mdd = np.nanmean(all_hist[:,:,MDD_ind,d],axis=2).T
plt.imshow(this_plot_mdd - this_plot_hc,cmap='bwr',vmin=vmin,vmax=vmax)
plt.yticks(np.arange(len(vel_edges))-0.5,labels_vel,fontsize=10)
plt.xticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.ylabel('d(CS)/dt')
plt.xlabel('CS: scene - face evidence',fontsize=15)
plt.title('Day %i' % d)
plt.colorbar()
plt.show()
#plt.subplot(1,3,3)
plt.figure(figsize=(20,10))
d=2
this_plot_hc = np.nanmean(all_hist[:,:,HC_ind,d],axis=2).T
this_plot_mdd = np.nanmean(all_hist[:,:,MDD_ind,d],axis=2).T
plt.imshow(this_plot_mdd-this_plot_hc,cmap='bwr',vmin=vmin,vmax=vmax)
plt.yticks(np.arange(len(vel_edges))-0.5,labels_vel,fontsize=10)
plt.xticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.ylabel('d(CS)/dt')
plt.xlabel('CS: scene - face evidence',fontsize=15)
plt.title('Day %i' % d)
plt.colorbar()
plt.show()
#plt.show()
# now plot for subject
s = 0
subject_key = 'subject' + str(subjects[s])
fig,ax = plt.subplots()
for d in np.arange(nDays):
    day_key = 'day' + str(d)
    #plt.subplot(1,3,d+1)
#   plt.figure()
    pos = all_x[subject_key][day_key][0]
    vel = all_dx[subject_key][day_key][0]
    #plt.scatter(x,y,alpha=0.5,s = 1.5,color='k')
    sns.jointplot(x=x,y=y,kind='kde',space=0,color='b')
    # pos_range = np.linspace(-1,1,100)
    # vel_range = np.linspace(-2,2,100)
    # posv,velv = np.meshgrid(pos_range,vel_range)
    # pos_deriv = vel[0:-1]
    # vel_deriv = np.diff(vel)
    # plt.quiver(posv,velv,pos_deriv,vel_deriv,alpha=0.75)

    plt.xlim([-1,1])
    plt.ylim([-2,2])
    plt.xlabel('categ sep')
    plt.ylabel('d(categ sep)/dt')
plt.show()

# do 3 separate things for each day
d = 0
mdd_x = []
mdd_dx = []
hc_x = []
hc_dx = []

for s in np.arange(nsubs):
    subject_key = 'subject' + str(subjects[s])
    day_key = 'day' + str(d)
    x = all_x[subject_key][day_key][0]
    y = all_dx[subject_key][day_key][0]
    if subjects[s] < 100:
        hc_x.extend(x)
        hc_dx.extend(y)
    if subjects[s] > 100:
        mdd_x.extend(x.flatten())
        mdd_dx.extend(y)

xnew = hc_x.copy()
xnew.extend(mdd_x)
dxnew = hc_dx.copy()
dxnew.extend(mdd_dx)
data = {}
data['x'] = xnew
data['dx'] = dxnew
subjects = np.ones((len(xnew)))
subjects[0:len(hc_x)] = 0
data['groups'] = subjects
df = pd.DataFrame.from_dict(data)


g = sns.jointplot(x=hc_x,y=hc_dx,kind='kde',space=0,color='k',ratio=3)
#plt.plot([-100,100],[0,0],color='k')
#plt.plot([0,0],[-100,100],color='k')
#g.xlim([-1.5,1.5])
#g.ylim([-1.5,1.5])
plt.show()

g = sns.jointplot(x=mdd_x,y=mdd_dx,kind='kde',space=0,color='r',ratio=3)
#plt.plot([-100,100],[0,0],color='k')
#plt.plot([0,0],[-100,100],color='k')
#g.xlim([-1.5,1.5])
#g.ylim([-1.5,1.5])
plt.show()


i=0
g = sns.JointGrid("x", "dx", df)

for x,group_data in df.groupby("groups"):
    if i==0:
        color = 'k'
        label = 'HC'
    else:
        color = 'r'
        label = 'MDD'
    sns.kdeplot(group_data['x'], ax=g.ax_marg_x, legend=False,color=color)
    sns.kdeplot(group_data['dx'], ax=g.ax_marg_y, vertical=True, legend=False,color=color)
    #g.ax_joint.plot(group_data['x'],group_data['dx'], "o", ms=1,color=color)
    #g.ax_joint.plot(group_data['x'],group_data['dx'], alpha=0.5,color=color)
    #g.ax_joint.scatter(group_data['x'],group_data['dx'],alpha=0.5,color=color)
    #g.ax_joint.kdeplot(group_data['x'],group_data['dx'],alpha=0.5,color=color)
    #g.plot_joint(sns.kdeplot)
    g.plot_joint(sns.distplot,x=group_data['x'],y=group_data['dx'])
    i += 1
plt.show()


# CHANGE COLORMAP
hc = df[df.groups==0]
mdd = df[df.groups==1]
ax = sns.kdeplot(hc.x, hc.dx,cmap="Greys", shade=False, shade_lowest=False,alpha=1,gridsize=100)
ax = sns.kdeplot(mdd.x, mdd.dx, cmap="Greens", shade=False, shade_lowest=False,alpha=1,gridsize=100)
# ax = sns.jointplot(hc.x, hc.dx, kind="hex", cmap="Greys")
# ax = sns.jointplot(mdd.x, mdd.dx, kind="hex", color="#4CB391")
ax.set_xlim([-1.4,1.4])
ax.set_ylim([-0.35,0.35])
ax.set_xlabel('category separation')
ax.set_ylabel('d(x)/dt')
plt.show()


g = sns.JointGrid("x", "dx", df)
sns.kdeplot(x='x',y='dx', ax=g.ax_marg_x, legend=False,data=df)
sns.kdeplot(day_tips["tip"], ax=g.ax_marg_y, vertical=True, legend=False)
    g.ax_joint.plot(day_tips["total_bill"], day_tips["tip"], "o", ms=5)


tips = sns.load_dataset("tips")
g = sns.JointGrid("total_bill", "tip", tips)
for x, group_data in df.groupby("groups"):
    sns.kdeplot(group_data['x'], ax=g.ax_marg_x, legend=False)
    sns.kdeplot(group_data['dx'], ax=g.ax_marg_y, vertical=True, legend=False)
    g.ax_joint.plot(group_data['x'],group_data['dx'], "o", ms=1)
plt.show()
tips = sns.load_dataset("tips")

for d in np.arange(nDays):
    fig,ax = plt.subplots()
    for s in np.arange(nsubs):
        subject_key = 'subject' + str(subjects[s])
        day_key = 'day' + str(d)
        if subjects[s] < 100:
            plt.subplot(1,2,1)
        elif subjects[s] > 100:
            plt.subplot(1,2,2)
        x = all_x[subject_key][day_key][0]
        y = all_dx[subject_key][day_key][0]
        #plt.scatter(x,y,alpha=0.5,s = 1.5,color='k')
        plt.plot(x,y,lw=1,alpha=0.1,color='k')
        plt.plot([-100,100],[0,0],color='k')
        plt.plot([0,0],[-100,100],color='k')
        plt.xlim([-1.5,1.5])
        plt.ylim([-2,2])
        plt.xlabel('categ sep')
        plt.ylabel('d(categ sep)/dt')
    plt.show()

fig,ax = plt.subplots()
for d in np.arange(nDays):
    plt.subplot(1,3,d+1)
    for s in np.arange(nsubs):
        subject_key = 'subject' + str(subjects[s])
        day_key = 'day' + str(d)
        if subjects[s] < 100:
            color='k'
        elif subjects[s] > 100:
            color='r'
        x = all_x[subject_key][day_key][0]
        y = all_dx[subject_key][day_key][0]
        plt.scatter(x,y,alpha=0.1,s = 1,color=color)
        plt.plot([-100,100],[0,0],color='k')
        plt.plot([0,0],[-100,100],color='k')
        plt.xlim([-1.5,1.5])
        plt.ylim([-2,2])
        plt.xlabel('categ sep')
        plt.ylabel('d(categ sep)/dt')
plt.show()

plt.figure()
plt.plot(x,label='x')
plt.plot(y,  label='dx')
plt.legend()
plt.show()

plt.figure()
#plt.plot(x,y, '-o', lw=10)
plt.scatter(x,y,alpha=0.5,s = 1.5)
plt.plot([-100,100],[0,0],color='k')
plt.plot([0,0],[-100,100],color='k')
plt.xlim([-1.5,1.5])
plt.ylim([-2,2])
plt.xlabel('categ sep')
plt.ylabel('d(categ sep)/dt')
plt.show()
# everything is 0 for some reason :()
# plot by run 
s=0
d=0
alpha=0.5
fig,ax = plt.subplots(1,3,sharey='col')
d=0
for s in np.arange(nsubs):
    if s in HC_ind:
        color = 'k'
    elif s in MDD_ind:
        color = 'r'
    ax[0].plot(neg_dec[s,:,d],color=color,alpha=alpha)
    ax[0].set_xlim([0,d1_runs-1])
hc_mean = np.nanmean(neg_dec[HC_ind,0:d1_runs,d],axis=0)
mdd_mean = np.nanmean(neg_dec[MDD_ind,0:d1_runs,d],axis=0)
ax[0].plot(hc_mean,color='k',lw=5,label='MDD')
ax[0].plot(mdd_mean,color='r',lw=5,label='MDD')

d=1
for s in np.arange(nsubs):
    if s in HC_ind:
        color = 'k'
    elif s in MDD_ind:
        color = 'r'
    ax[1].plot(neg_dec[s,:,d],color=color,alpha=alpha)
    ax[1].set_xlim([0,d2_runs-1])
hc_mean = np.nanmean(neg_dec[HC_ind,0:d2_runs,d],axis=0)
mdd_mean = np.nanmean(neg_dec[MDD_ind,0:d2_runs,d],axis=0)
ax[1].plot(hc_mean,color='k',lw=5,label='MDD')
ax[1].plot(mdd_mean,color='r',lw=5,label='MDD')

d=2
for s in np.arange(nsubs):
    if s in HC_ind:
        color = 'k'
    elif s in MDD_ind:
        color = 'r'
    ax[2].plot(neg_dec[s,:,d],color=color,alpha=alpha)
    ax[2].set_xlim([0,d3_runs-1])
hc_mean = np.nanmean(neg_dec[HC_ind,0:d3_runs,d],axis=0)
mdd_mean = np.nanmean(neg_dec[MDD_ind,0:d3_runs,d],axis=0)
ax[2].plot(hc_mean,color='k',lw=5,label='MDD')
ax[2].plot(mdd_mean,color='r',lw=5,label='MDD')

plt.show()

# average over day
neg_dec_day = np.nanmean(neg_dec,axis=1)
stat = neg_dec_day
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
addComparisonStat(p/2,0,2,np.nanmax(stat),0.03)
plt.ylabel('p(decrease|negative)')
plt.xticks(np.arange(3))
plt.show()

mdd_mean = np.mean(neg_dec_day[MDD_ind,:],axis=0)
hc_mean = np.mean(neg_dec_day[HC_ind,:],axis=0)
mdd_err = scipy.stats.sem(neg_dec_day[MDD_ind,:],axis=0)
hc_err = scipy.stats.sem(neg_dec_day[HC_ind,:],axis=0)
alpha=0.3
plt.figure()
for s in np.arange(nsubs):
    if s in HC_ind:
        color = 'k'
    elif s in MDD_ind:
        color = 'r'
    plt.plot(neg_dec_day[s,:],'-',ms=10,color=color,alpha=alpha,lw=2)
plt.errorbar(x=np.arange(nDays),y=hc_mean,yerr=hc_err,color='k',lw=2,label='MDD',fmt='-o',ms=10)
plt.errorbar(x=np.arange(nDays),y=mdd_mean,yerr=mdd_err,color='r',lw=2,label='MDD',fmt='-o',ms=10)
plt.xlabel('day')
plt.ylabel('p(decrease| negative)')
plt.xticks([0,1,2])
plt.show()

# difference between day 1 and day 3?
scipy.stats.ttest_rel(neg_dec_day[MDD_ind,0],neg_dec_day[MDD_ind,2])
scipy.stats.ttest_rel(neg_dec_day[HC_ind,0],neg_dec_day[HC_ind,2])

# does this change from day 3 to day 1 predict MADRS at the end?

neg_dec_day = np.nanmean(neg_area_all,axis=1)
stat = neg_dec_day
fig = plotPosterStyle(stat,subjects)
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.03)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(stat),0.03)
x,y = nonNan(stat[HC_ind,2],stat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,2,np.nanmax(stat),0.03)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat(p/2,0,2,np.nanmax(stat),1)

plt.ylabel('Area under -0.1')
plt.xticks(np.arange(3))
plt.show()



mdd_mean = np.mean(neg_dec_day[MDD_ind,:],axis=0)
hc_mean = np.mean(neg_dec_day[HC_ind,:],axis=0)
mdd_err = scipy.stats.sem(neg_dec_day[MDD_ind,:],axis=0)
hc_err = scipy.stats.sem(neg_dec_day[HC_ind,:],axis=0)
alpha=0.3
plt.figure()
for s in np.arange(nsubs):
    if s in HC_ind:
        color = 'k'
    elif s in MDD_ind:
        color = 'r'
    plt.plot(neg_dec_day[s,:],'-',ms=10,color=color,alpha=alpha,lw=2)
plt.errorbar(x=np.arange(nDays),y=hc_mean,yerr=hc_err,color='k',lw=2,label='MDD',fmt='-o',ms=10)
plt.errorbar(x=np.arange(nDays),y=mdd_mean,yerr=mdd_err,color='r',lw=2,label='MDD',fmt='-o',ms=10)
plt.xlabel('day')
plt.ylabel('area under -0.1')
plt.xticks([0,1,2])
plt.show()

# difference between day 1 and day 3?
scipy.stats.ttest_rel(neg_dec_day[MDD_ind,0],neg_dec_day[MDD_ind,2])
scipy.stats.ttest_rel(neg_dec_day[HC_ind,0],neg_dec_day[HC_ind,2])

colors=['k','r']
M = getMADRSscoresALL()
fig = plt.figure(figsize=(10,7))
for s in np.arange(nsubs):
    subjectNum  = subjects[s]
    this_sub_madrs = M[subjectNum]
    madrs_change = this_sub_madrs[1] - this_sub_madrs[0]
    if subjectNum < 100:
        style = 0
    elif subjectNum > 100:
        style = 1
    plt.plot(neg_dec_day[s,2] - neg_dec_day[s,0],madrs_change,marker='.',ms=20,color=colors[style],alpha=0.5)
plt.xlabel('p(decrease|negative) Change 3 - 1')
plt.ylabel('MADRS Change 3 - 1')
plt.show()
#scipy.stats.pearsonr(neg_dec_day[MDD_ind,2] - neg_dec_day[MDD_ind,0])
