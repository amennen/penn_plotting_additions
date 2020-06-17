# purpose of function: read in amygdala contrasts and get average for each session, run for subjec

# %matplotlib inline
# import IPython
# from IPython import get_ipython
# get_ipython().magic('matplotlib inline')
import scipy
import matplotlib
import matplotlib.pyplot as plt
#from nilearn import image
#from nilearn.input_data import NiftiMasker
#from nilearn import plotting
import nibabel
#from nilearn.masking import apply_mask
#from nilearn.image import load_img
#from nilearn.image import new_img_like
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, svm, metrics
from sklearn.linear_model import Ridge
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.feature_selection import SelectFwe
from scipy import signal
from scipy.fftpack import fft, fftshift
from scipy import interp
import statistics
params = {'legend.fontsize': 'large',
          'figure.figsize': (5, 3),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
font = {'weight': 'normal',
        'size': 22}
plt.rc('font', **font)
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, f_classif, GenericUnivariateSelect, SelectKBest, chi2
from sklearn.feature_selection import RFE
import os
#import seaborn as sns
import pandas as pd
from anne_additions.plotting_pretty.commonPlotting import *

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

def transition_matrix_shift(transitions,nstates,nshift):
    n=nstates
    M = [[0]*n for _ in range(n)]
    for (i,j) in zip(transitions,transitions[nshift:]):
        if i < 0 or j < 0:
            pass
        else:
          M[i][j] += 1
    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

def find_max_mode(list1):
    list_table = statistics._counts(list1)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(list1)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
            if 0 not in new_list and len(new_list) > 1:
              print('********')
              print(new_list)
              print('********')
              max_mode = 0
            else:
              max_mode = max(new_list) # use the max value here
    return max_mode



results = '/data/jux/cnds/amennen/rtAttenPenn/gazedata/all_states.mat'
subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101,102,103,104,105, 106,107,108,109,110,111,112,113,114])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
d = scipy.io.loadmat(results,struct_as_record=False)
states = d['all_states']
ndays = 3 # for now do 3 days --np.shape(ratios)[3]
nsamples = np.shape(states)[2]
ntrials = np.shape(states)[1]
nsubjects = np.shape(states)[0]
# shape of ratios: (7, 12, 4, 3)
# n subjects x 12 trials x 4 emotions x 3 days

# now plot all averages
DYSPHORIC = 1;
THREAT = 2;
NEUTRAL = 3;
POSITIVE = 4;
emotions = ['DYSPHORIC', 'THREAT', 'NEUTRAL', 'POSITIVE']
emo = ['DYS', 'THR', 'NEU', 'POS']
# 30 second trials --> 120 Hz or 120 samples/second in the trial
# look 3 seconds later --> 120 * 3 = 360 shift
# states_list = [x[x!=0] for x in np.split(this_state, np.where(this_state==0)[0]) if len(x[x!=0])]
# for istate in np.arange(len(states_list)):
#   print(states_list[istate][0])

nstates=4
f_old=120
f_new = 15 # how many samples/second you want
samples_to_bin = np.int(f_old/f_new)
n_samples_total = f_old*30
n_downsampled = f_new*30
nshift = int(3*f_new) # how many samples to shift ahead

all_matrices = np.zeros((nstates,nstates,nsubjects,ndays))
p_state = np.zeros((nstates,nsubjects,ndays))
for s in np.arange(nsubjects):
  print(s)
  for d in np.arange(ndays):
    print(d)
    this_day_matrices = np.zeros((nstates,nstates,ntrials))
    for t in np.arange(ntrials):
      this_state = states[s,t,:,d]

      new_gaze = np.zeros((n_downsampled,))
      for g in np.arange(n_downsampled):
        sample_1 = g*samples_to_bin
        sample_2 = (g+1)*samples_to_bin
        all_pts = this_state[sample_1:sample_2]
        new_gaze[g] = find_max_mode(all_pts)

      for st in np.arange(nstates):
        n_this_gaze = len(np.argwhere(new_gaze==(st+1)))
        n_nonzero = len(np.argwhere(new_gaze>0))
        if n_this_gaze > 0:
          p_state[st,s,d] = n_this_gaze/n_nonzero
        
      # ORIGINAL WAY - just remove nonzero points
      #values = this_state[this_state!=0]
      values_0ind = (new_gaze.copy() - 1).astype(int)
      print(t)
      # check that all states have been gotten to - if not add zero row/col
      values_taken = np.unique(values_0ind)
      values_taken = values_taken[values_taken>=0]
      if len(values_taken) == nstates:
        this_day_matrices[:,:,t] = np.array(transition_matrix_shift(values_0ind,nstates,nshift))
        for st in np.arange(nstates):
          # check that they actually transitioned
          row = this_day_matrices[st,:,t]
          if np.sum(row) == 0: # never actually transitioned to any state
            this_day_matrices[st,:,t] = np.nan

        #print(this_day_matrices[:,:,t])
      elif len(values_taken) == nstates -1: # only 1 state wasn't visited
        temp_transition_matrix = np.array(transition_matrix_shift(values_0ind,nstates,nshift))
        other = [x for x in np.arange(nstates) if x not in values_taken]
        this_day_matrices[:,:,t] = np.nan
        this_day_matrices[values_taken[0],:,t] = temp_transition_matrix[values_taken[0],:]
        this_day_matrices[values_taken[1],:,t] = temp_transition_matrix[values_taken[1],:]
        this_day_matrices[values_taken[2],:,t] = temp_transition_matrix[values_taken[2],:]
      elif len(values_taken) == 1: # just looked at 1 the whole time
      # they literally were only in one state the entire trial so one 1 on the diagonal
        only_one = values_taken[0]
        this_day_matrices[:,:,t] = np.nan
        this_day_matrices[only_one,only_one,t] = 1
        others = [x for x in np.arange(nstates) if x != only_one]
        this_day_matrices[only_one,others,t] = 0
      elif len(values_taken) == 2:
        print('only 2')
        others  = [x for x in np.arange(nstates) if x not in values_taken]
        temp_transition_matrix = np.array(transition_matrix_shift(values_0ind,nstates,nshift))
        this_day_matrices[:,:,t] = np.nan
        this_day_matrices[values_taken[0],:,t] = temp_transition_matrix[values_taken[0],:]
        this_day_matrices[values_taken[1],:,t] = temp_transition_matrix[values_taken[1],:]
      elif len(np.unique(values_0ind)) == 0:
        this_day_matrices[:,:,t] = np.nan
      # if the shifting makes things weird
      bad_vals = np.argwhere(np.nansum(this_day_matrices[:,:,t],axis=1) == 0)
      if len(bad_vals) > 0:
        for b in np.arange(len(bad_vals)):
          this_day_matrices[bad_vals[b],:,t] = np.nan

    # take average over all trials
    all_matrices[:,:,s,d] = np.nanmean(this_day_matrices,axis=2)
    print(np.sum(all_matrices[:,:,s,d],axis=1))

stat = all_matrices[0,0,:,:]
fig,ax = plotPosterStyle_DF(stat,subjects)
plt.xticks(np.arange(3),('Pre NF', 'Mid NF', 'Post NF'))
plt.ylabel('p(dys|dys)')
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.01)
x,y = nonNan(stat[HC_ind,2],stat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,2,np.nanmax(stat),0.01)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat(p/2,0,2,np.nanmax(stat),0.05)
#plt.set_xticklabels(('Pre NF','Mid NF' 'Post NF'))
plt.show()

# for t in np.arange(ntrials):
#   print(this_day_matrices[:,:,t])
#   print(np.nansum(this_day_matrices[:,:,t],axis=1))
# plt.plot(this_state)
# plt.show()
# plt.plot(values)
# plt.show()

# # bin by number of seconds

# new_gaze = np.zeros((n_downsampled,))
# for g in np.arange(n_downsampled):
#   sample_1 = g*samples_to_bin
#   sample_2 = (g+1)*samples_to_bin
#   all_pts = this_state[sample_1:sample_2]
#   new_gaze[g] = find_max_mode(all_pts)

# plt.plot(np.linspace(0,len(this_state)/f_old,len(this_state)),this_state)
# plt.plot(np.linspace(0,len(new_gaze)/f_new,len(new_gaze)),new_gaze ,'.', ms=10)
# plt.show()


# plt.plot(values)
# plt.show()

## TO DO: FIX SO THEY ALL SUM TO 1
i=0
j=0
# i,j 0.2 is for -1-->-0.9
# try to stay positive = 15 15
data_mat = all_matrices[i,j,:,:]

fig = plotPosterStyle(data_mat,subjects)
#plt.ylim([0,1])
x,y = nonNan(data_mat[MDD_ind,0],data_mat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat(p/2,0,2,np.nanmax(data_mat),0.05)
x,y = nonNan(data_mat[HC_ind,0],data_mat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(data_mat),0.01)
x,y = nonNan(data_mat[HC_ind,2],data_mat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,2,np.nanmax(data_mat),0.01)
plt.show()



stat = p_state[0,:,:]
fig,ax = plotPosterStyle_DF(stat,subjects)
plt.xticks(np.arange(3),('Pre NF', 'Mid NF', 'Post NF'))
plt.ylabel('p(dys)')
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.01)
x,y = nonNan(stat[HC_ind,2],stat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,2,np.nanmax(stat),0.01)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat(p/2,0,2,np.nanmax(stat),0.05)
#plt.set_xticklabels(('Pre NF','Mid NF' 'Post NF'))
plt.show()


# plot difference of MADRS vs. others
M = getMADRSscoresALL()
d1,d2,d3 = getMADRSdiff(M,subjects)
all_neg_change = data_mat[:,2] - data_mat[:,0]

colors = ['k', 'r'] # HC, MDD
fig = plt.figure(figsize=(10,7))
for s in np.arange(nsubjects):
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
x,y = nonNan(-1*d1[MDD_ind],all_neg_change[MDD_ind])
scipy.stats.pearsonr(x,y)

i=3
j=3
# i,j 0.2 is for -1-->-0.9
# try to stay positive = 15 15
data_mat = all_matrices[i,j,:,:]

fig = plotPosterStyle(data_mat,subjects)
#plt.ylim([0,1])
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


color_d3_pts = np.array([252 ,174 ,145])/255
color_d3_avg = np.array([203 ,24 ,29])/255
color_d1_pts = np.array([204,204,204])/255
color_d1_avg = np.array([82,82,82])/255
ind_bin = 0
nbins=nstates
labels_pos = emotions
#plt.figure()
fig,ax = plt.subplots()
for s in np.arange(nsubjects):
    if s in HC_ind:
        plt.subplot(1,2,1)
        matrix1 = all_matrices[ind_bin,:,s,0]
        matrix3 = all_matrices[ind_bin,:,s,2]
        plt.plot(np.arange(nbins),matrix1,color=color_d1_pts,alpha=0.7)
        plt.plot(np.arange(nbins),matrix3,color=color_d3_pts,alpha=0.7)
        plt.xticks(np.arange(nbins),labels_pos,fontsize=8)
        plt.xlabel('value B')
        plt.ylabel('p(B|dysphoric)')
        #plt.ylabel('p(B|happy)')
        #plt.legend()
    elif s in MDD_ind:
        plt.subplot(1,2,2)
        matrix1 = all_matrices[ind_bin,:,s,0]
        matrix3 = all_matrices[ind_bin,:,s,2]
        plt.plot(np.arange(nbins),matrix1,color=color_d1_pts,alpha=0.7)
        plt.plot(np.arange(nbins),matrix3,color=color_d3_pts,alpha=0.7)
        plt.xticks(np.arange(nbins),labels_pos,fontsize=8)
        plt.xlabel('value B')
        #plt.ylabel('p(B|dysphoric)')
plt.subplot(1,2,1)
hc_m1_avg = np.nanmean(all_matrices[ind_bin,:,HC_ind,0],axis=0)
hc_m3_avg = np.nanmean(all_matrices[ind_bin,:,HC_ind,2],axis=0)
plt.plot(np.arange(nbins),hc_m1_avg,color=color_d1_avg,alpha=1,lw=5,label='day 1')
plt.plot(np.arange(nbins),hc_m3_avg,color=color_d3_avg,alpha=1, lw=5,label='day 3')
plt.title('HC Group')
plt.ylim([0,1])
plt.legend()
plt.subplot(1,2,2)
mdd_m1_avg = np.nanmean(all_matrices[ind_bin,:,MDD_ind,0],axis=0)
mdd_m3_avg = np.nanmean(all_matrices[ind_bin,:,MDD_ind,2],axis=0)
plt.plot(np.arange(nbins),mdd_m1_avg,color=color_d1_avg,alpha=1,lw=5,label='day 1')
plt.plot(np.arange(nbins),mdd_m3_avg,color=color_d3_avg,alpha=1, lw=5,label='day 3')
plt.title('MDD Group')
plt.ylim([0,1])

plt.legend()
plt.show()





# now plot
vmin=0
vmax=1
#day 1 first plt.figure(figsize=(20,20))
d=0
# make sequential colormap
fig,ax = plt.subplots(figsize=(20,20))
plt.subplot(1,3,1)
this_plot_hc = np.nanmean(all_matrices[:,:,HC_ind,d],axis=2)
plt.imshow(this_plot_hc,cmap='Reds',vmin=vmin,vmax=vmax)
plt.colorbar()
plt.yticks(np.arange(nbins),labels_pos,fontsize=8)
plt.xticks(np.arange(nbins),labels_pos,fontsize=8)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('HC',fontsize=20)
plt.subplot(1,3,2)
this_plot_mdd = np.nanmean(all_matrices[:,:,MDD_ind,d],axis=2)
plt.yticks(np.arange(nbins),labels_pos,fontsize=8)
plt.xticks(np.arange(nbins),labels_pos,fontsize=8)
plt.imshow(this_plot_mdd,cmap='Reds',vmin=vmin,vmax=vmax)
plt.xlabel('value B')
plt.title('MDD',fontsize=20)
plt.colorbar()
plt.show()
# to do: understand output of transition matrix -- which is A/which is B, plot for each day
plt.figure(figsize=(10,10))
plt.imshow(this_plot_mdd-this_plot_hc,cmap='bwr',vmin=-.1,vmax=.1) # for half, max diff is .2, for all days, max diff is 0.1
plt.yticks(np.arange(nbins),labels_pos,fontsize=8)
plt.xticks(np.arange(nbins),labels_pos,fontsize=8)
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
plt.yticks(np.arange(nbins),labels_pos,fontsize=8)
plt.xticks(np.arange(nbins),labels_pos,fontsize=8)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('HC2 - HC0')
plt.colorbar()
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(this_plot_mdd3-this_plot_mdd1,cmap='bwr',vmin=-.11,vmax=.11) # for half, max diff is .11, for all days, max diff is 0.06
plt.yticks(np.arange(nbins),labels_pos,fontsize=8)
plt.xticks(np.arange(nbins),labels_pos,fontsize=8)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('MDD2 - MDD0')
plt.colorbar()
plt.show()




