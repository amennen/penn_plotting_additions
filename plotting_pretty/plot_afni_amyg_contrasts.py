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

params = {'legend.fontsize': 'large',
          'figure.figsize': (5, 3),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
font = {'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
## Frame wise displacement isn't here
fmriprep_out="/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/fmriprep"
task_path = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/faces'
run_path = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/normalized_runs'
cf_dir = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/fsl/first_level/confound_EVs'
timing_path = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/timing_files'
analyses_out = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/stats'
whole_brain_mask = '/data/jux/cnds/amennen/rtAttenPenn/MNI_things/mni_icbm152_t1_tal_nlin_asym_09c_BOLD_mask_Penn.nii'
amygdala_mask = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/mni_anat/LAMYG_in_MNI_overlapping.nii.gz'

allsubjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101,102,103,104,105,106,107,108,109,110,111,112])
HC_ind = np.argwhere(allsubjects<100)[:,0]
MDD_ind = np.argwhere(allsubjects>100)[:,0]
nsubs = len(allsubjects)
all_subject_averages_fear_neutral = np.zeros((nsubs,2))

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
        output_text = "{0}/{1}_{2}_task-faces_negminusneut_amgyavg.txt".format(output_path,bids_id,ses_id)
        f = open(output_text,"r") #opens file with name of "test.txt"
        z = f.readline()
        all_subject_averages_fear_neutral[s,ses] = float(z[:-1])


# (1) Is there a difference before NF?

colors = ['k', 'r'] # HC, MDD

fig = plt.figure(figsize=(10,7))
# plot for each subject
DAY = 0
for s in np.arange(nsubs):
    if allsubjects[s] < 100:
        style = 0
        index=1
        plt.plot(index,all_subject_averages_fear_neutral[s,DAY],marker='.', ms=20,color=colors[style],alpha=0.5)
    else:
        style = 1
        index=0
        plt.plot(index,all_subject_averages_fear_neutral[s,DAY], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.array([1]),np.mean(all_subject_averages_fear_neutral[HC_ind,DAY],axis=0),lw = 2,marker="+",ms=20,color=colors[0],yerr=scipy.stats.sem(all_subject_averages_fear_neutral[HC_ind,DAY],axis=0), label='HC')
plt.errorbar(np.array([0]),np.mean(all_subject_averages_fear_neutral[MDD_ind,DAY],axis=0),lw = 2,marker="+",ms=20,color=colors[1],yerr=scipy.stats.sem(all_subject_averages_fear_neutral[MDD_ind,DAY],axis=0), label='MDD')
plt.xticks(np.arange(2),('MDD', 'HC'))
plt.ylim([-.3,.3])
plt.xlabel('Group')
plt.ylabel('Amygdala Activity: Fear > Neutral')
plt.title('Amygdala activity Visit 1')
plt.legend()
fig=plt.figure(figsize=(10,7))
DAY = 1
for s in np.arange(nsubs):
    if allsubjects[s] < 100:
        style = 0
        index=1
        plt.plot(index,all_subject_averages_fear_neutral[s,DAY],marker='.', ms=20,color=colors[style],alpha=0.5)
    else:
        style = 1
        index=0
        plt.plot(index,all_subject_averages_fear_neutral[s,DAY], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.array([1]),np.mean(all_subject_averages_fear_neutral[HC_ind,DAY],axis=0),lw = 2,marker="+",ms=20,color=colors[0],yerr=scipy.stats.sem(all_subject_averages_fear_neutral[HC_ind,DAY],axis=0), label='HC')
plt.errorbar(np.array([0]),np.mean(all_subject_averages_fear_neutral[MDD_ind,DAY],axis=0),lw = 2,marker="+",ms=20,color=colors[1],yerr=scipy.stats.sem(all_subject_averages_fear_neutral[MDD_ind,DAY],axis=0), label='MDD')
plt.xticks(np.arange(2),('MDD', 'HC'))
plt.ylim([-.3,.3])
plt.xlabel('Group')
plt.ylabel('Amygdala Activity: Fear > Neutral')
plt.title('Amygdala activity Visit 2')
plt.legend()
plt.show()

f,p= scipy.stats.ttest_ind(all_subject_averages_fear_neutral[MDD_ind,0],all_subject_averages_fear_neutral[HC_ind,0])

time_diff = np.diff(all_subject_averages_fear_neutral)
fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
    if allsubjects[s] < 100:
        style = 0
        index=1
        plt.plot(index,time_diff[s],marker='.', ms=20,color=colors[style],alpha=0.5)
    else:
        style = 1
        index=0
        plt.plot(index,time_diff[s], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.array([1]),np.mean(time_diff[HC_ind],axis=0),lw = 2,marker="+",ms=20,color=colors[0],yerr=scipy.stats.sem(time_diff[HC_ind],axis=0), label='HC')
plt.errorbar(np.array([0]),np.mean(time_diff[MDD_ind],axis=0),lw = 2,marker="+",ms=20,color=colors[1],yerr=scipy.stats.sem(time_diff[MDD_ind],axis=0), label='MDD')
plt.xticks(np.arange(2),('MDD', 'HC'))
plt.xlabel('Group')
plt.ylabel('Amygdala Activity: Fear > Neutral')
plt.title('Amygdala activity  Post - Pre')
plt.legend()
plt.show()
f,p= scipy.stats.ttest_ind(time_diff[MDD_ind],time_diff[HC_ind])
finalp = p/2 # for one-sided t-test for MDD to go down more than HC
print('f is %2.2f' % f)
print('one sided t-test is %2.2f'  % finalp)
