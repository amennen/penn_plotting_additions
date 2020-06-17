# purpose: average amygdala activity for each subject, compile and plot

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

allsubjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101,102,103,104,105,106,107,108,109,110,111,112,113,114])
HC_ind = np.argwhere(allsubjects<100)[:,0]
MDD_ind = np.argwhere(allsubjects>100)[:,0]
nsubs = len(allsubjects)

ntrials = 6
all_subject_averages_neutral = np.zeros((nsubs,ntrials,2))
all_subject_averages_fearful = np.zeros((nsubs,ntrials,2))
all_subject_averages_happy = np.zeros((nsubs,ntrials,2))


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
        # get all neutral first
        for trial in np.arange(ntrials):
            output_text = "{0}/{1}_{2}_task-faces_neutral_{3}_amgyavg.txt".format(output_path,bids_id,ses_id,trial)
            f = open(output_text,"r") 
            z = f.readline()
            f.close()
            all_subject_averages_neutral[s,trial,ses] = float(z[:-1])
        # now get all negative
        for trial in np.arange(ntrials):
            output_text = "{0}/{1}_{2}_task-faces_fearful_{3}_amgyavg.txt".format(output_path,bids_id,ses_id,trial)
            f = open(output_text,"r") 
            z = f.readline()
            f.close()
            all_subject_averages_fearful[s,trial,ses] = float(z[:-1])
        # now get all happy
        for trial in np.arange(ntrials):
            output_text = "{0}/{1}_{2}_task-faces_happy_{3}_amgyavg.txt".format(output_path,bids_id,ses_id,trial)
            f = open(output_text,"r") 
            z = f.readline()
            f.close()
            all_subject_averages_happy[s,trial,ses] = float(z[:-1])



stat = all_subject_averages_neutral
# just plot over all day first
fig = plotPosterStyle_multiplePTS(stat,allsubjects)
plt.show()

stat = all_subject_averages_fearful
# just plot over all day first
fig = plotPosterStyle_multiplePTS(stat,allsubjects)
plt.show()


stat = all_subject_averages_happy
# just plot over all day first
fig = plotPosterStyle_multiplePTS(stat,allsubjects)
plt.show()

# do first minus last
fearful_diff_end = all_subject_averages_fearful[:,5,:] - all_subject_averages_neutral[:,5,:]
fearful_diff_beg = all_subject_averages_fearful[:,0,:] - all_subject_averages_neutral[:,0,:]

last_half = np.mean(all_subject_averages_fearful[:,3:6,:],axis=1) 
first_half = np.mean(all_subject_averages_fearful[:,0:3,:],axis=1) 
total = np.mean(all_subject_averages_fearful[:,0:6,:],axis=1)


stat = last_half - first_half
fig,ax = plotPosterStyle_DF(stat,allsubjects)
plt.xticks(np.arange(2),('Pre NF', 'Post NF'))
plt.show()

stat = last_half - first_half
#stat = total
fig,ax = plotPosterStyle_DF(stat,allsubjects)
plt.xticks(np.arange(2),('Pre NF', 'Post NF'))
plt.ylabel('average beta')
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.01)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(stat),0.01)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,1])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat(p/2,0,1,np.nanmax(stat),0.05)
plt.show()

stat = total
fig = plotPosterStyle(stat,allsubjects)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,1])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat(p/2,0,1,np.nanmax(stat),0.05)
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.03)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(stat),0.03)
plt.ylabel('happy > neutral')
plt.xticks(np.arange(2),('Pre NF', 'Post NF'))
#plt.title('A''ignore_neutralF - A''ignore_sadF')
plt.show()

M = getMADRSscoresALL()
d1,d2,d3 = getMADRSdiff(M,allsubjects)
all_neg_change = stat[:,1] - stat[:,0]
#all_neg_change = last_half[:,1] - last_half[:,0]
colors = ['k', 'r'] # HC, MDD
fig = plt.figure(figsize=(10,7))
for s in np.arange(nsubs):
  subjectNum  = allsubjects[s]
  madrs_change = d2[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(all_neg_change[s],d2[s],marker='.',ms=20,color=colors[style],alpha=0.5)
plt.xlabel('Decrease in amygdala activity 3 - 1')
plt.ylabel('Decrease in MADRS: V5 -> V1')
#plt.xlim([-0.4,0.4])
plt.show()
x,y = nonNan(d1[MDD_ind],all_neg_change[MDD_ind])
scipy.stats.pearsonr(x,y)
x,y = nonNan(d2[MDD_ind],all_neg_change[MDD_ind])
scipy.stats.pearsonr(x,y)
x,y = nonNan(-1*d1,all_neg_change)
scipy.stats.pearsonr(x,y)


stat = np.mean(all_subject_averages_neutral[:,3:6,:],axis=1) - np.mean(all_subject_averages_neutral[:,0:3,:],axis=1) 
fig = plotPosterStyle(stat,allsubjects)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,1])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat(p/2,0,1,np.nanmax(stat),0.05)
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.03)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(stat),0.03)
plt.ylabel('happy > neutral')
plt.xticks(np.arange(2),('Pre NF', 'Post NF'))
#plt.title('A''ignore_neutralF - A''ignore_sadF')
plt.show()