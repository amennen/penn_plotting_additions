# purpose: average amygdala activity for each subject, compile and plot
# go through checks that nick suggestion on 1/7 meeting

import os
import glob
from shutil import copyfile
import pandas as pd
import json
import numpy as np
from subprocess import call
import sys
import scipy.stats
import nibabel as nib
import nilearn
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
import seaborn as sns
import pandas as pd
from anne_additions.plotting_pretty.commonPlotting import *

params = {'legend.fontsize': 'large',
          'figure.figsize': (5, 3),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
font = {'weight': 'normal',
        'size': 22}
plt.rc('font', **font)


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

## Frame wise displacement isn't here
fmriprep_out="/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/fmriprep"
task_path = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/faces'
run_path = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/normalized_runs'
cf_dir = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/fsl/first_level/confound_EVs'
timing_path = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/timing_files'
analyses_out = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/stats'
whole_brain_mask = '/data/jux/cnds/amennen/rtAttenPenn/MNI_things/mni_icbm152_t1_tal_nlin_asym_09c_BOLD_mask_Penn.nii'
amygdala_mask = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/mni_anat/LAMYG_in_MNI_overlapping.nii.gz'

# do for each of the clusters
allsubjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101,102,103,104,105,106,107,108,109,110,111,112,113,114])
HC_ind = np.argwhere(allsubjects<100)[:,0]
MDD_ind = np.argwhere(allsubjects>100)[:,0]
nsubs = len(allsubjects)

ntrials = 2
nclusters=13
### LEFT OFF HERE -- right code to load in all the data and create plots -- see if the same trends are shown on day 1
all_subject_averages_fearful_contrast = np.zeros((nsubs,nclusters,ntrials,2))
all_subject_averages_happy_contrast = np.zeros((nsubs,nclusters,ntrials,2))
all_subject_averages_object_contrast = np.zeros((nsubs,nclusters,2))
# make into subjects x days x emotions
all_subject_averages_fearful = np.zeros((nsubs,nclusters,ntrials,2))
all_subject_averages_neutral = np.zeros((nsubs,nclusters,ntrials,2))

for s in np.arange(len(allsubjects)):
    subjectNum = allsubjects[s]
    bids_id = 'sub-{0:03d}'.format(subjectNum)

    # concatenate confound EVS
    print(bids_id)
    sessions = [1,3]
    for ses in np.arange(len(sessions)):
        subjectDay = sessions[ses]
        ses_id = 'ses-{0:02d}'.format(subjectDay)
        print(ses_id)
        output_path = "{0}/{1}/{2}".format(analyses_out,bids_id,ses_id)
        # now get all negative
        for trial in np.arange(ntrials):
          # if trial==0:
          for cluster in np.arange(nclusters):
            output_text = "{0}/{1}_{2}_task-faces_f_m_n_{3}_half_amgyavg_ALL_OPTIONS_cluster{4}.txt".format(output_path,bids_id,ses_id,trial,cluster)
            # elif trial == 1: # whoops made a typo
            #   output_text = "{0}/{1}_{2}_task-faces_f_m_n_{3}_half_amgyavg_ALL_OPTIONS.txt".format(output_path,bids_id,ses_id,trial)
            f = open(output_text,"r") 
            z = f.readline()
            f.close()
            all_subject_averages_fearful_contrast[s,cluster,trial,ses] = float(z[:-1])
        # now get all happy contrast
        for trial in np.arange(ntrials):
          for cluster in np.arange(nclusters):
            output_text = "{0}/{1}_{2}_task-faces_h_m_n_{3}_half_amgyavg_ALL_OPTIONS_cluster{4}.txt".format(output_path,bids_id,ses_id,trial,cluster)
            f = open(output_text,"r") 
            z = f.readline()
            f.close()
            all_subject_averages_happy_contrast[s,cluster,trial,ses] = float(z[:-1])
        for trial in np.arange(ntrials):
          for cluster in np.arange(nclusters):
            output_text = "{0}/{1}_{2}_task-faces_fearful_{3}_half_amgyavg_ALL_OPTIONS_cluster{4}.txt".format(output_path,bids_id,ses_id,trial,cluster)
            f = open(output_text,"r") 
            z = f.readline()
            f.close()
            all_subject_averages_fearful[s,cluster,trial,ses] = float(z[:-1])
        for trial in np.arange(ntrials):
          for cluster in np.arange(nclusters):
            output_text = "{0}/{1}_{2}_task-faces_neutral_{3}_half_amgyavg_ALL_OPTIONS_cluster{4}.txt".format(output_path,bids_id,ses_id,trial,cluster)
            f = open(output_text,"r") 
            z = f.readline()
            f.close()
            all_subject_averages_neutral[s,cluster,trial,ses] = float(z[:-1])
        # now get object contrast
        for cluster in np.arange(cluster):
          output_text = "{0}/{1}_{2}_task-faces_f_m_o_half_amgyavg_ALL_OPTIONS_cluster{3}.txt".format(output_path,bids_id,ses_id,cluster)
          f = open(output_text,"r") 
          z = f.readline()
          f.close()
          all_subject_averages_object_contrast[s,cluster,ses] = float(z[:-1])


# check for specific cluster
amyg = 9
# redo the plot Yvette showed on each day
# dotted is negative
# solid is neutral
colors_dark = ['#636363','#de2d26']
colors_light = ['#636363','#de2d26']
fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
plt.subplot(1,2,1)
day=0

x = np.arange(ntrials)
y = all_subject_averages_neutral[HC_ind,amyg,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC neutral')
y = all_subject_averages_fearful[HC_ind,amyg,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[0],label='HC negative')

y = all_subject_averages_neutral[MDD_ind,amyg,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD neutral')
y = all_subject_averages_fearful[MDD_ind,amyg,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[1],label='MDD negative')
x,y = nonNan(all_subject_averages_fearful[MDD_ind,amyg,1,day],all_subject_averages_fearful[HC_ind,amyg,1,day])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,.9,1.1,0.2,0.05,0,'MDD > HC ')
plt.legend()

plt.subplot(1,2,2)
sns.despine()
day=1
x = np.arange(ntrials)
y = all_subject_averages_neutral[HC_ind,amyg,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC neutral')
y = all_subject_averages_fearful[HC_ind,amyg,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[0],label='HC negative')

y = all_subject_averages_neutral[MDD_ind,amyg,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD neutral')
y = all_subject_averages_fearful[MDD_ind,amyg,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[1],label='MDD negative')
plt.legend()
plt.show()

# specifically plot contrast
stat = all_subject_averages_fearful_contrast[:,amyg,:,:]
diff_stat = stat[:,1,:] - stat[:,0,:]
fig = plotPosterStyle_multiplePTS(stat,allsubjects)
plt.subplot(1,2,1)
plt.ylim([-.6,.6])
plt.xticks(np.array([0,1]),('start','end'),fontsize=30)
plt.xlabel('time in block',fontsize=40)
plt.ylabel(r'LA($\beta$) neg > neutral',fontsize=40)
x,y = nonNan(diff_stat[MDD_ind,0],diff_stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.,1,0.4,0.05,0,'MDD increase > HC increase')
x,y = nonNan(stat[MDD_ind,0,0],stat[HC_ind,0,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.1,.1,0.2,0.05,0,'MDD < HC ')
x,y = nonNan(stat[MDD_ind,1,0],stat[HC_ind,1,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,.9,1.1,0.2,0.05,0,'MDD > HC ')
plt.subplot(1,2,2)
plt.ylim([-.6,.6])
plt.xticks(np.array([0,1]),('start','end'),fontsize=30)
plt.xlabel('time in block',fontsize=40)
plt.show()

PHG=11
fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
plt.subplot(1,2,1)
day=0

x = np.arange(ntrials)
y = all_subject_averages_neutral[HC_ind,PHG,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC neutral')
y = all_subject_averages_fearful[HC_ind,PHG,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[0],label='HC negative')

y = all_subject_averages_neutral[MDD_ind,PHG,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD neutral')
y = all_subject_averages_fearful[MDD_ind,PHG,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[1],label='MDD negative')
plt.legend()
plt.subplot(1,2,2)
sns.despine()
day=1
x = np.arange(ntrials)
y = all_subject_averages_neutral[HC_ind,PHG,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC neutral')
y = all_subject_averages_fearful[HC_ind,PHG,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[0],label='HC negative')

y = all_subject_averages_neutral[MDD_ind,PHG,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD neutral')
y = all_subject_averages_fearful[MDD_ind,PHG,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[1],label='MDD negative')
plt.legend()
plt.show()

# TO DO:  make arranged properly in the y axis - LOOK AT NEGATIVE FACES
stat = all_subject_averages_fearful_contrast
diff_stat = all_subject_averages_fearful_contrast[:,1,:] - all_subject_averages_fearful_contrast[:,0,:]
# just plot over all day first
fig = plotPosterStyle_multiplePTS(stat,allsubjects)
plt.subplot(1,2,1)
plt.ylim([-.55,.8])
plt.xticks(np.array([0,1]),('start','end'),fontsize=30)
plt.xlabel('time in block',fontsize=40)
plt.ylabel(r'LA response ($\beta$)',fontsize=40)
labels_pos_v = np.array([-.5,0,0.5])
labels_pos = labels_pos_v.astype(np.str)
plt.yticks(labels_pos_v,labels_pos,fontsize=30)
x,y = nonNan(diff_stat[MDD_ind,0],diff_stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.,1.,np.nanmax(stat),0.05,0,'MDD increase > HC increase')
x,y = nonNan(stat[MDD_ind,0,0],stat[HC_ind,0,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.1,.1,np.nanmax(stat)+.2,0.05,0,'MDD < HC ')
x,y = nonNan(stat[MDD_ind,1,0],stat[HC_ind,1,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,.9,1.1,np.nanmax(stat)+.2,0.05,0,'MDD > HC ')
plt.subplot(1,2,2)
plt.xticks(np.array([0,1]),('start','end'),fontsize=30)
plt.xlabel('time in block',fontsize=40)
plt.ylim([-.55,.8])
#plt.ylabel(r'$L amygdala \beta$',fontsize=40)
x,y = nonNan(diff_stat[MDD_ind,1],diff_stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0,1,np.nanmax(stat),0.05,0,' ')
labels_pos_v = np.array([-.5,0,0.5])
labels_pos = labels_pos_v.astype(np.str)
plt.yticks(labels_pos_v,labels_pos,fontsize=30)
#plt.savefig('poster_plots/amygdala_repsponse.png')
plt.show()


# NOW HAPPY FACES
stat = all_subject_averages_happy_contrast
diff_stat = all_subject_averages_happy_contrast[:,1,:] - all_subject_averages_happy_contrast[:,0,:]
# just plot over all day first
fig = plotPosterStyle_multiplePTS(stat,allsubjects)
plt.subplot(1,2,1)
plt.ylim([-.55,.7])
plt.xticks(np.array([0,1]),('start','end'),fontsize=30)
plt.xlabel('time in block',fontsize=40)
plt.ylabel(r'LA response ($\beta$)',fontsize=40)
labels_pos_v = np.array([-.5,0,0.5])
labels_pos = labels_pos_v.astype(np.str)
plt.yticks(labels_pos_v,labels_pos,fontsize=30)
x,y = nonNan(diff_stat[MDD_ind,0],diff_stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.,1.,np.nanmax(stat),0.05,0,'MDD increase > HC increase')

x,y = nonNan(stat[MDD_ind,0,0],stat[HC_ind,0,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.1,.1,np.nanmax(stat)+.2,0.05,0,'MDD < HC ')
x,y = nonNan(stat[MDD_ind,1,0],stat[HC_ind,1,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,.9,1.1,np.nanmax(stat)+.2,0.05,0,'MDD > HC ')

plt.subplot(1,2,2)
plt.xticks(np.array([0,1]),('start','end'),fontsize=30)
plt.xlabel('time in block',fontsize=40)
plt.ylim([-.55,.7])
#plt.ylabel(r'$L amygdala \beta$',fontsize=40)
x,y = nonNan(diff_stat[MDD_ind,1],diff_stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0,1,np.nanmax(stat),0.05,0,' ')
labels_pos_v = np.array([-.5,0,0.5])
labels_pos = labels_pos_v.astype(np.str)
plt.yticks(labels_pos_v,labels_pos,fontsize=30)
#plt.savefig('poster_plots/amygdala_repsponse.png')
plt.show()


# check objects
stat = all_subject_averages_object_contrast[:,amyg,:]
#stat = total
fig,ax = plotPosterStyle_DF(stat,allsubjects)
plt.xticks(np.arange(2),('Pre NF', 'Post NF'))
plt.ylabel(r'LA negative face $\Delta\beta$')
# x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
# t,p = scipy.stats.ttest_ind(x,y)
# addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,'$MDD_1 > HC_1$')
# x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
# t,p = scipy.stats.ttest_ind(x,y)
# addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05)
# x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,1])
# t,p = scipy.stats.ttest_rel(x,y)
# addComparisonStat_SYM(p/2,0.2,1.2,np.nanmax(stat)+.03,0.1,'$MDD_1 > MDD_3$')
#plt.ylim([-.45,1])
plt.xlabel('')
plt.show()


### IS IMPROVEMENT RELATED TO LESS REACTIVITY??
amyg = 9
stat = all_subject_averages_fearful[:,amyg,1,:] - all_subject_averages_fearful[:,amyg,0,:]
M = getMADRSscoresALL()
d1,d2,d3 = getMADRSdiff(M,allsubjects)
all_neg_change = stat[:,1] - stat[:,0]
data_mat = all_matrices[0,0,:,:]
rt_change = data_mat[:,2] - data_mat[:,0]

#all_neg_change = last_half[:,1] - last_half[:,0]
colors = ['k', 'r'] # HC, MDD
fig = plt.figure(figsize=(10,7))
for s in np.arange(nsubs):
  subjectNum  = allsubjects[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(-1*all_neg_change[s],-1*d1[s],marker='.',ms=20,color=colors[style],alpha=0.5)
  #plt.plot(-1*all_neg_change[s],-1*rt_change[s],marker='.',ms=20,color=colors[style],alpha=0.5)

plt.xlabel('Improvement in reactivity: Post - Pre')
plt.ylabel('Improvement in depression severity: MADRS: V5 -> V1')
#plt.xlim([-0.4,0.4])
plt.show()
x,y = nonNan(d1[MDD_ind],-1*all_neg_change[MDD_ind])
#x,y = nonNan(rt_change[MDD_ind],-1*all_neg_change[MDD_ind])
scipy.stats.pearsonr(x,y)
x,y = nonNan(d2[MDD_ind],all_neg_change[MDD_ind])
scipy.stats.pearsonr(x,y)
x,y = nonNan(-1*d1,all_neg_change)
scipy.stats.pearsonr(x,y)
