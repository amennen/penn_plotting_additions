
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

all_categories = ['fearful','happy', 'neutral', 'object']

n_beta_per_category = 9
nclusters=13
all_subject_averages_neutral = np.zeros((nsubs,n_beta_per_category,nclusters+1,2))
all_subject_averages_fearful = np.zeros((nsubs,n_beta_per_category,nclusters+1,2))
all_subject_averages_happy = np.zeros((nsubs,n_beta_per_category,nclusters+1,2))
all_subject_averages_object = np.zeros((nsubs,n_beta_per_category,nclusters+1,2))

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
        for category in all_categories:
          for n in np.arange(n_beta_per_category):
            for cluster in np.arange(nclusters+1): # last one is amygdala
              if cluster < nclusters:
                output_text = "{0}/{1}_{2}_task-faces_{3}_{4}_TENT_cluster{5}.txt".format(output_path,bids_id,ses_id,category,n,cluster)
              else:
                output_text = "{0}/{1}_{2}_task-faces_{3}_{4}_TENT_LAMYG_overlapping.txt".format(output_path,bids_id,ses_id,category,n)
              f = open(output_text,"r") 
              z = f.readline()
              f.close()
              if category == 'fearful':
                all_subject_averages_fearful[s,n,cluster,ses] = float(z[:-1])
              elif category == 'happy':
                all_subject_averages_happy[s,n,cluster,ses] = float(z[:-1])
              elif category == 'neutral':
                all_subject_averages_neutral[s,n,cluster,ses] = float(z[:-1])
              elif category == 'object':
                all_subject_averages_object[s,n,cluster,ses] = float(z[:-1])


colors_dark = ['#636363','#de2d26']
colors_light = ['#636363','#de2d26']
# first check: own amygdala fearful and neutral, day 1
# maybe smooth more??
cluster=13
day=0
fig,ax = plt.subplots(figsize=(17,9))
sns.despine()

x = np.arange(n_beta_per_category)
y = all_subject_averages_neutral[HC_ind,:,cluster,day]
# for s in np.arange(len(HC_ind)):
#   plt.plot(x,y[s,:])
# #plt.ylim([-2.5,2.25])
# plt.show()
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC neutral')
y = all_subject_averages_fearful[HC_ind,:,cluster,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[0],label='HC negative')
y = all_subject_averages_neutral[MDD_ind,:,cluster,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD neutral')
y = all_subject_averages_fearful[MDD_ind,:,cluster,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[1],label='MDD negative')
#plt.xlim([1,10])
#plt.ylim([-1,1])
plt.legend()
plt.show()
        