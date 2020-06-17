# purpose of function: read in amygdala contrasts and get average for each session, run for subjec

# %matplotlib inline
# import IPython
# from IPython import get_ipython
# get_ipython().magic('matplotlib inline')
import scipy
import matplotlib
import matplotlib.pyplot as plt
import itertools
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


def consecutive(data,indices, stepsize=1):
    fixations = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    if np.sum(indices) != -1:
      indices = np.split(indices, np.where(np.diff(data) != stepsize)[0]+1)
    else:
      indices = np.nan
    return fixations,indices

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
        if i > nstates or j > nstates:
          if i < nstates and j > nstates:
            # in this case, you're in a state and then you go to no data 
            # in this case, look within window as long as it's within seconds that you're looking
            # how to basically find the next station?
            # problem - because you always saccade to another one first
            pass
          else:
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
        max_mode = max(new_list) # use the max value here
    return max_mode



results = '/data/jux/cnds/amennen/rtAttenPenn/gazedata/all_fixations.mat'
subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101,102,103,104,105, 106,107,108,109,110,111,112,113,114])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
d = scipy.io.loadmat(results,struct_as_record=False)
states = d['all_fixations'] # this is nsubjects x 12 trials x 3600 trials x 4 days
ndays = 3 # for now do 3 days --np.shape(ratios)[3]
nsamples = np.shape(states)[2]
ntrials = np.shape(states)[1]
nsubjects = np.shape(states)[0]
ndays = np.shape(states)[3]

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
# every 600 samples is new segment
nsamples_per_seg=600
nsamples_per_sec=120
nemotions = len(emo)
nseg=6
# calculate disengagement time: from start how long until you go until next image
# would need to know index of all fixations starts
ratio_per_seg = np.zeros((nsubjects,ntrials,nseg,ndays,nemotions))*np.nan
dys_first = np.zeros((nsubjects,ntrials,nseg,ndays,nemotions))*np.nan
fixation_durations = np.zeros((nsubjects,ntrials,nseg,ndays,nemotions))*np.nan
fixation_durations_across_trial = np.zeros((nsubjects,ntrials,ndays,nemotions))*np.nan
fixation_durations_dys_first = np.zeros((nsubjects,ntrials,nseg,ndays,nemotions))*np.nan
first_orientation = np.zeros((nsubjects,ntrials,ndays,nemotions)) *np.nan
total_viewing_time = np.zeros((nsubjects,ntrials,ndays,nemotions)) *np.nan
dysphoric_segment_probability=np.zeros((nsubjects,ntrials,nseg-1,ndays))*np.nan
time_to_go_next_image = np.zeros((nsubjects,ntrials,ndays,nemotions)) * np.nan
options = np.arange(nemotions)
for s in np.arange(nsubjects):
  for d in np.arange(ndays):
    for t in np.arange(ntrials):
      all_trial_indices = np.arange(nsamples)
      trial_ts = states[s,t,:,d]
      fixations = trial_ts[trial_ts>0]
      fixations_trial_indices = all_trial_indices[trial_ts>0]
      if len(fixations): # looked somewhere
        orientation = fixations[0] - 1 #subtract 1 for matlab-python indexing
        other_options = options[options!=orientation]
        first_orientation[s,t,d,orientation] = 1 
        first_orientation[s,t,d,other_options] = 0
        n_points_recorded = len(fixations)
        # get consecutive parts across all trial
        c_trial,indices_trial = consecutive(fixations,fixations_trial_indices,stepsize=0)
        # PROBLEM: if super far away in time, should count as separate fixations?
        nfixations = len(c_trial)
        # calculate average fixation duration for that emotion each tmie
        emotion_count = np.zeros((nemotions,))
        emotion_sum = np.zeros((nemotions,))*np.nan
        for f in np.arange(nfixations):
          if not np.any(np.diff(indices_trial[f])>1):
            this_start = indices_trial[f][0]
            if f < nfixations-1:
              next_start = indices_trial[f+1][0]
            else:
              next_start = indices_trial[f][-1]
            this_emotion = c_trial[f][0] - 1
            emotion_count[this_emotion] += 1
            t_diff = (next_start-this_start)
            if emotion_count[this_emotion] ==1:
              emotion_sum[this_emotion] = t_diff
            else:
             emotion_sum[this_emotion] = emotion_sum[this_emotion] + t_diff
        time_to_go_next_image[s,t,d,:] = emotion_sum/emotion_count*(1/n_points_recorded)
        for e in np.arange(nemotions):
          # calculate total viewing time
          # first check if there was a fixation for that emotion
          total_viewing_time[s,t,d,e] = len(np.argwhere(fixations==e+1))/n_points_recorded
          ### should it be 0 or nan if they attended somewhere but not to that category in that trial?
          fixations_this_emotion = [i for i in np.arange(nfixations) if c_trial[i][0]==e+1]
          if np.any(fixations==e+1):
            avg_fixations = np.zeros((len(fixations_this_emotion,)))
            avg_time_next_image = np.zeros((len(fixations_this_emotion,)))
            for f in np.arange(len(fixations_this_emotion)):
              avg_fixations[f] = len(c_trial[fixations_this_emotion[f]])/nsamples_per_sec
            fixation_durations_across_trial[s,t,d,e] = np.mean(avg_fixations)
        for seg in np.arange(nseg):
          this_seg_ts=trial_ts[seg*nsamples_per_seg:(seg+1)*nsamples_per_seg]
          nonzero_ts = this_seg_ts[this_seg_ts>0]
          c = consecutive(nonzero_ts,-1,stepsize=0)
          # all_fixation_values = [c[i][0] for i in np.arange(nfixations)]
          # nfixations = len(c)
          #flattened_fixations = list(itertools.chain(*c))
          # have a check where if subjects have consecutive separate fixations, merge them
          #fixations_check = consecutive(flattened_fixations,stepsize=0)
          fixations_check = c[0]
          nfixations = len(fixations_check)
          seg_fixations = np.zeros((nfixations,nemotions))*np.nan

          # have a check where if subjects have consecutive separate fixations, merge them
          if len(fixations_check) >1:
            for f in np.arange(nfixations):
              this_fixation = fixations_check[f]
              #duration = len(this_fixation)/nsamples_per_sec
              # NEW: NORMALIZE BY TIME FIXATED IN SEGMENT - LATER CHANGE TO WHOLE TRIAL
              duration = len(this_fixation)/len(np.argwhere(this_seg_ts>0))
              seg_fixations[f,this_fixation[0]-1] = duration # python-matlab indexing difference
          #average over all fixations in that segment

          for e in np.arange(nemotions):
            ratio_per_seg[s,t,seg,d,e] = len(np.argwhere(this_seg_ts==e+1))/nsamples_per_seg
            if len(c) > 1:
              fixation_durations[s,t,seg,d,e] = np.nanmean(seg_fixations[:,e])
        if ratio_per_seg[s,t,0,d,0] > 0 :
          # given that we looked at dysphoric in the first section, what is the probability
          # that we'll still be looking at dysphoric in each of the next segments
          dys_first[s,t,:,d,:] = ratio_per_seg[s,t,:,d,:]
          fixation_durations_dys_first[s,t,:,d,:] = fixation_durations[s,t,:,d,:]
          for seg2 in np.arange(1,nseg):
            if ratio_per_seg[s,t,seg2,d,0] > 0:
              dysphoric_segment_probability[s,t,seg2-1,d] = 1
            else:
              dysphoric_segment_probability[s,t,seg2-1,d] = 0
# NEW 1/8: given attending to dysphoric in this segment, how likely are you i shifted later
shift_probability = np.zeros((nsubjects,ntrials,nseg-1,ndays))*np.nan
for s in np.arange(nsubjects):
  for d in np.arange(ndays):
    for t in np.arange(ntrials):
      counter_shifts = np.zeros((nseg,))
      counter_shifts_check = np.zeros((nseg,))
      dysphoric_fixations = ratio_per_seg[s,t,:,d,0] # will be nan if didn't look anywhere in that trial
      # will be 0 if looked somehwere but not dysphoric
      # so we check if greater than 0 to be sure that they looked at dysphoric first

      for seg1 in np.arange(1):
        if dysphoric_fixations[seg1]>0: # given they looked at dysphoric in that segment
          # how often did they look at another
          for seg2 in np.arange(seg1+1,nseg):
            shift = seg2-seg1
            # check that they looked somewhere else
            # all_fixations_in_seg = ratio_per_seg[s,t,seg2,d,:]
            # if np.any(all_fixations_in_seg):
            counter_shifts_check[shift] +=1
            if dysphoric_fixations[seg2]>0:
              # yes added 1
              counter_shifts[shift] += 1
      if counter_shifts_check[1] > 0:
        shift_probability[s,t,:,d] = counter_shifts[1:]/counter_shifts_check[1:]
trial_averages=np.nanmean(shift_probability,axis=1)
shift_prob_improvement = trial_averages[:,1,2] - trial_averages[:,1,0]
stat=trial_averages


M = getMADRSscoresALL()
d1,d2,d3 = getMADRSdiff(M,subjects)
data_mat = all_matrices[0,0,:,:]
all_neg_change = data_mat[:,2] - data_mat[:,0]
colors = ['k', 'r'] # HC, MDD
colors = ['#636363','#de2d26']
#fig = plt.figure(figsize=(10,7))
fig,ax = plt.subplots(figsize=(12,10))
sns.despine()
for s in np.arange(nsubs):
  subjectNum  = subjects[s]
  madrs_change = d2[s]
  gaze_change = shift_prob_improvement[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(-1*shift_prob_improvement[s],-1*d1[s],marker='.',ms=30,color=colors[style])
plt.xlabel('improvement in gaze stickiness',fontsize=40)
plt.ylabel('improvement in MADRS',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim([-.5,.8])
plt.show()

fig,ax = plt.subplots(figsize=(12,10))
sns.despine()
for s in np.arange(nsubs):
  subjectNum  = subjects[s]
  madrs_change = d2[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(-1*all_neg_change[s],-1*shift_prob_improvement[s],marker='.',ms=30,color=colors[style])
plt.ylabel('improvement in gaze stickiness',fontsize=40)
plt.xlabel('improvement in neg stickiness',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim([-.5,.8])
plt.show()

x,y = nonNan(-1*all_neg_change[MDD_ind],-1*shift_prob_improvement[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)

text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.47,20,text, ha='left',va='top',color='k',fontsize=25)
x,y = nonNan(-1*all_neg_change[MDD_ind],-1*d1[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.47,15,text, ha='left',va='top',color='k',fontsize=25)
labels_pos_v = np.array([-0.4,0,0.4,0.8])
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(labels_pos_v,labels_pos,fontsize=30)

plt.show()


fig=plotPosterStyle_multiplePTS(stat[:,:,np.array([0,2,3])],subjects)
plt.subplot(1,3,1)
plt.ylim([0,1])
x,y = nonNan(stat[MDD_ind,1,0],stat[HC_ind,1,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,1,1,0.6,0,0,text_above=[])
plt.title('Pre NF')
plt.xticks(np.arange(5),np.arange(5)+1)
plt.ylabel('$p(dysphoric_i|dypshoric_0)$')
plt.xlabel('segment')
plt.subplot(1,3,2)
plt.title('Post NF')
plt.xticks(np.arange(5),np.arange(5)+1)
plt.ylim([0,1])
plt.xlabel('segment')
plt.subplot(1,3,3)
plt.title('1M FU')
plt.xticks(np.arange(5),np.arange(5)+1)
plt.ylim([0,1])
plt.xlabel('segment')
plt.show()

# plot probability per emotion per segment like kellough paper - ratio time attending in each segment
trial_average_ratio_per_seg = np.nanmean(ratio_per_seg,axis=1)
stat = trial_average_ratio_per_seg[:,:,:,POSITIVE-1]
fig=plotPosterStyle_multiplePTS(stat[:,:,np.array([0,2,3])],subjects)
plt.show()
# plot per day, on each day
nonzero_ts = this_seg_ts[this_seg_ts>0]
nfixations = len(c)

# get day 0 average over all subjects in each group
trial_averages = np.nanmean(ratio_per_seg,axis=1)
trial_averages = np.nanmean(dysphoric_segment_probability,axis=1)
# any_next_prob = dysphoric_segment_probability
# any_next_prob[any_next_prob>1] = 1
# trial_averages = np.nanmean(any_next_prob,axis=1)
# stat=trial_averages
# fig,ax = plotPosterStyle_DF(stat[:,np.array([0,2,3])],subjects)
# plt.show()
# trial_averages = np.nanmean(any_next_prob,axis=1)
stat=trial_averages
x,y = nonNan(stat[MDD_ind,1,0],stat[HC_ind,1,0])
t,p = scipy.stats.ttest_ind(x,y)

x,y = nonNan(stat[MDD_ind,1,0],stat[MDD_ind,1,1])
t,p = scipy.stats.ttest_rel(x,y)

fig=plotPosterStyle_multiplePTS(stat[:,:,np.array([0,2,3])],subjects)
plt.subplot(1,3,1)
plt.ylim([0,1])
plt.title('Pre NF')
plt.xticks(np.arange(5),np.arange(5)+1)
plt.ylabel('$p(dysphoric_i|dypshoric_0)$')
plt.xlabel('segment')
plt.subplot(1,3,2)
plt.title('Post NF')
plt.xticks(np.arange(5),np.arange(5)+1)
plt.ylim([0,1])
plt.xlabel('segment')
plt.subplot(1,3,3)
plt.title('1M FU')
plt.xticks(np.arange(5),np.arange(5)+1)
plt.ylim([0,1])
plt.xlabel('segment')
plt.show()
trial_averages = np.nanmean(dys_first,axis=1)
#trial_averages = np.nanmean(fixation_durations,axis=1)
#trial_averages = np.nanmean(fixation_durations_dys_first,axis=1)

HC_avg = np.nanmean(trial_averages[HC_ind,:,0,:],axis=0)
MDD_avg = np.nanmean(trial_averages[MDD_ind,:,0,:],axis=0)
fig,ax = plt.subplots()
plt.subplot(1,2,1)
for e in np.arange(nemotions):
  plt.plot(HC_avg[:,e],label=emo[e])
plt.legend()
plt.ylim([0,1])
plt.subplot(1,2,2)
for e in np.arange(nemotions):
  plt.plot(MDD_avg[:,e],label=emo[e])
plt.ylim([0,1])
plt.show()

### LOOK AT INITIAL ORIENTATION
trial_average_first_orientation=(np.nanmean(first_orientation,axis=1))
trial_averages = np.nanmean(first_orientation[:,:,:,DYSPHORIC-1],axis=1)
stat= trial_averages
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
plt.ylim([0,1])
plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))

x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,1.8,2.2,np.nanmax(stat),0.05,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,2.2,np.nanmax(stat)+.2,0.05,0,'$MDD_1 > MDD_5$')

x,y = nonNan(stat[HC_ind,3],stat[MDD_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,2.8,3.2,np.nanmax(stat),0.05,0,'$MDD > HC$')

plt.ylabel('ratio initial orientation - dysphoric')
plt.show()

trial_averages = np.nanmean(first_orientation[:,:,:,POSITIVE-1],axis=1)
stat = trial_averages
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
plt.ylim([0,1])
plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))

x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,1.8,2.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,2.2,np.nanmax(stat)+.2,0.05,0,'$MDD_1 < MDD_5$')

x,y = nonNan(stat[HC_ind,3],stat[MDD_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,2.8,3.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

plt.ylabel('ratio initial orientation - positive')
plt.show()

emo_diff = trial_average_first_orientation[:,:,POSITIVE-1] - trial_average_first_orientation[:,:,DYSPHORIC-1]
stat = emo_diff
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
plt.ylim([0,1])
plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))

x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,1.8,2.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,2.2,np.nanmax(stat)+.2,0.05,0,'$MDD_1 < MDD_5$')

x,y = nonNan(stat[HC_ind,3],stat[MDD_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,2.8,3.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

plt.ylabel('ratio initial orientation: positive - dysphoric')
plt.show()



##### NOW LOOK AT FIXATION DURATIONS
## average over all segments and trials for that day
# first average over all segments in a given trial
#segment_average_duration = np.nanmean(fixation_durations,axis=2)
trial_average_total_viewing = np.nanmean(total_viewing_time,axis=1)
stat = trial_average_total_viewing[:,:,DYSPHORIC-1]
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
plt.ylim([0,.9])
plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))

x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,1.8,2.2,np.nanmax(stat),0.05,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,2.2,np.nanmax(stat)+.2,0.05,0,'$MDD_1 > MDD_5$')

x,y = nonNan(stat[HC_ind,3],stat[MDD_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,2.8,3.2,np.nanmax(stat),0.05,0,'$MDD > HC$')

plt.ylabel('mean ratio total time - dysphoric')
plt.show()


stat = trial_average_total_viewing[:,:,POSITIVE-1]
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
plt.ylim([0,.9])
plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))

x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,1.8,2.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,2.2,np.nanmax(stat)+.2,0.05,0,'$MDD_1 < MDD_5$')

x,y = nonNan(stat[HC_ind,3],stat[MDD_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,2.8,3.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

plt.ylabel('mean ratio total time - positive')
plt.show()

emo_diff = trial_average_total_viewing[:,:,POSITIVE-1] - trial_average_total_viewing[:,:,DYSPHORIC-1] 
#emo_diff = trial_average_maintenance[:,:,DYSPHORIC-1] -trial_average_maintenance[:,:,POSITIVE-1]
stat=emo_diff
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
#plt.ylim([0,.9])
plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))

x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,1.8,2.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,2.2,np.nanmax(stat)+.2,0.05,0,'$MDD_1 < MDD_5$')

x,y = nonNan(stat[HC_ind,3],stat[MDD_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,2.8,3.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

plt.ylabel('mean ratio total time: positive - dysphoric')
plt.show()


#### NOW LOOK AT MAINTENANCE TIME ON IMAGES
#fixation_durations = np.zeros((nsubjects,ntrials,nseg,ndays,nemotions))*np.nan
trial_average_maintenance = np.nanmean(fixation_durations_across_trial,axis=1)
stat = trial_average_maintenance[:,:,DYSPHORIC-1]
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
#plt.ylim([0,.9])
plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))

x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,1.8,2.2,np.nanmax(stat),0.05,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,2.2,np.nanmax(stat)+.4,0.05,0,'$MDD_1 > MDD_5$')

x,y = nonNan(stat[HC_ind,3],stat[MDD_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,2.8,3.2,np.nanmax(stat),0.05,0,'$MDD > HC$')

plt.ylabel('mean maintenance time (s) - dysphoric')
plt.show()

stat = trial_average_maintenance[:,:,POSITIVE-1]
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
#plt.ylim([0,.9])
plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))

x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,1.8,2.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,2.2,np.nanmax(stat)+.4,0.05,0,'$MDD_1 < MDD_5$')

x,y = nonNan(stat[HC_ind,3],stat[MDD_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,2.8,3.2,np.nanmax(stat),0.05,0,'$MDD < HC$')

plt.ylabel('mean maintenance time (s) - positive')
plt.show()


############################### NEW AFTER 1/7 MEETING:
average_segments = np.nanmean(fixation_durations,axis=2)
trial_average_maintenance = np.nanmean(average_segments,axis=1)
trial_average_first_orientation=(np.nanmean(first_orientation,axis=1))
trial_average_total_viewing = np.nanmean(total_viewing_time,axis=1)
trial_average_fixation_dur = np.nanmean(fixation_durations_across_trial,axis=1)

#all_data=trial_average_maintenance

all_data=trial_average_first_orientation
fig,ax=plotPosterStyle_DF_valence(all_data[:,np.array([0,2,3]),:],subjects,emo,'ratio first orientation')
# plt.subplot(1,2,1)
# plt.xticks(np.arange(3),('Pre NF','Post NF', '1M FU'))
# plt.subplot(1,2,2)
# plt.xticks(np.arange(3),('Pre NF','Post NF', '1M FU'))
plt.show()

all_data=trial_average_total_viewing
fig,ax=plotPosterStyle_DF_valence(all_data[:,np.array([0,2,3]),:],subjects,emo,'ratio viewing')
# plt.subplot(1,2,1)
# plt.xticks(np.arange(3),('Pre NF','Post NF', '1M FU'))
# plt.subplot(1,2,2)
# plt.xticks(np.arange(3),('Pre NF','Post NF', '1M FU'))
plt.show()

trial_average_time = np.nanmean(time_to_go_next_image,axis=1)
all_data=trial_average_time
fig,ax=plotPosterStyle_DF_valence(all_data[:,np.array([0,2,3]),:],subjects,emo,'time to change image')
# plt.subplot(1,2,1)
# plt.xticks(np.arange(3),('Pre NF','Post NF', '1M FU'))
# plt.subplot(1,2,2)
# plt.xticks(np.arange(3),('Pre NF','Post NF', '1M FU'))
plt.show()

all_data=trial_average_fixation_dur ## to do: put in check to make sure separate fixations are dealt as separate fixations even if the same emotion!!
fig,ax=plotPosterStyle_DF_valence(all_data[:,np.array([0,2,3]),:],subjects,emo,'fixation duration (s)')
# plt.subplot(1,2,1)
# plt.xticks(np.arange(3),('Pre NF','Post NF', '1M FU'))
# plt.subplot(1,2,2)
# plt.xticks(np.arange(3),('Pre NF','Post NF', '1M FU'))
plt.show()

nstates=4
nshift = 120 * 2
nshift = 120 *10
all_matrices = np.zeros((nstates,nstates,nsubjects,ndays))
for s in np.arange(nsubjects):
  print(s)
  for d in np.arange(ndays):
    print(d)
    this_day_matrices = np.zeros((nstates,nstates,ntrials))
    for t in np.arange(ntrials):
      this_state = states[s,t,:,d]
      states_list = [x[x!=0] for x in np.split(this_state, np.where(this_state==0)[0]) if len(x[x!=0])]
      n_vals = len(states_list)
      values = np.zeros((n_vals,))
      for istate in np.arange(n_vals):
        values[istate] = states_list[istate][0]
      # states_nan = this_state.copy()
      # states_nan[states_nan==0] = -100
      # states_nan = states_nan.copy() - 1
      # values_0ind = states_nan
      # ORIGINAL WAY - just remove nonzero points
      #values = this_state[this_state!=0]
      values_0ind = (values.copy() - 1).astype(int)
      b=np.append(values_0ind[1:],-1)
      consec_states=values_0ind[values_0ind!=b]
      values_0ind = consec_states
      print(t)
      # check that all states have been gotten to - if not add zero row/col
      values_taken = np.unique(values_0ind)
      values_taken = values_taken[values_taken<nstates]

      this_day_matrices[:,:,t] = np.array(transition_matrix(values_0ind,nstates))
      if len(values_taken) == nstates:
        this_day_matrices[:,:,t] = np.array(transition_matrix(values_0ind,nstates))
        for st in np.arange(nstates):
          # check that they actually transitioned
          row = this_day_matrices[st,:,t]
          if np.sum(row) == 0: # never actually transitioned to any state
            this_day_matrices[st,:,t] = np.nan

        #print(this_day_matrices[:,:,t])
      elif len(values_taken) == nstates -1: # only 1 state wasn't visited
        temp_transition_matrix = np.array(transition_matrix(values_0ind,nstates))
        other = [x for x in np.arange(nstates) if x not in values_taken]
        this_day_matrices[:,:,t] = np.nan
        this_day_matrices[values_taken[0],:,t] = temp_transition_matrix[values_taken[0],:]
        this_day_matrices[values_taken[1],:,t] = temp_transition_matrix[values_taken[1],:]
        this_day_matrices[values_taken[2],:,t] = temp_transition_matrix[values_taken[2],:]
      elif len(values_taken) == 1: # just looked at 1 the whole time
      # they literally were only in one state the entire trial so one 1 on the diagonal
        only_one = values_taken[0]
        this_day_matrices[:,:,t] = np.nan
        this_day_matrices[only_one,only_one,t] = 0#1
        others = [x for x in np.arange(nstates) if x != only_one]
        this_day_matrices[only_one,others,t] = 0
      elif len(values_taken) == 2:
        print('only 2')
        others  = [x for x in np.arange(nstates) if x not in values_taken]
        temp_transition_matrix = np.array(transition_matrix(values_0ind,nstates))
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

for t in np.arange(ntrials):
  print(this_day_matrices[:,:,t])
  print(np.nansum(this_day_matrices[:,:,t],axis=1))
plt.plot(this_state)
plt.show()
plt.plot(values)
plt.show()

# bin by number of seconds
f_old=120
f_new = 30 # how many samples/second you want
samples_to_bin = np.int(f_old/f_new)
n_samples_total = f_old*30
n_downsampled = f_new*30
new_gaze = np.zeros((n_downsampled,))
for g in np.arange(n_downsampled):
  sample_1 = g*samples_to_bin
  sample_2 = (g+1)*samples_to_bin
  all_pts = this_state[sample_1:sample_2]
  new_gaze[g] = find_max_mode(all_pts)

plt.plot(np.linspace(0,len(this_state)/f_old,len(this_state)),this_state)
plt.plot(np.linspace(0,len(new_gaze)/f_new,len(new_gaze)),new_gaze ,'.', ms=10)
plt.show()


plt.plot(values)
plt.show()

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
ind_bin = 3
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
        plt.ylabel('p(B|happy)')
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
nbins=nstates
labels_pos = emotions

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




