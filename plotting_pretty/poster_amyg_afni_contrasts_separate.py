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
# we want to average amygdla activity for:
# negative last - negagtive first --> 7
# negative last - neutral last --> repeated oh well -- 8
# negative first -  neutral first --> 10
# positive last - positive first --> 6
# postiive last - neutral last --> 12
# positive first -  neutral first --> 11
all_subject_averages_negL_negF = np.zeros((nsubs,2))
all_subject_averages_negL_neuL = np.zeros((nsubs,2))
all_subject_averages_negF_neuF = np.zeros((nsubs,2))
all_subject_averages_posL_posF = np.zeros((nsubs,2))
all_subject_averages_posL_neuL = np.zeros((nsubs,2))
all_subject_averages_posF_neuF = np.zeros((nsubs,2))

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

        output_text = "{0}/{1}_{2}_task-faces_negL_negF_amgyavg.txt".format(output_path,bids_id,ses_id)
        f = open(output_text,"r") 
        z = f.readline()
        f.close()
        all_subject_averages_negL_negF[s,ses] = float(z[:-1])

        output_text = "{0}/{1}_{2}_task-faces_negL_neutL_amgyavg.txt".format(output_path,bids_id,ses_id)
        f = open(output_text,"r") 
        z = f.readline()
        f.close()
        all_subject_averages_negL_neuL[s,ses] = float(z[:-1])

        output_text = "{0}/{1}_{2}_task-faces_negF_neutF_amgyavg.txt".format(output_path,bids_id,ses_id)
        f = open(output_text,"r") 
        z = f.readline()
        f.close()
        all_subject_averages_negF_neuF[s,ses] = float(z[:-1])

        output_text = "{0}/{1}_{2}_task-faces_posL_posF_amgyavg.txt".format(output_path,bids_id,ses_id)
        f = open(output_text,"r") 
        z = f.readline()
        f.close()
        all_subject_averages_posL_posF[s,ses] = float(z[:-1])

        output_text = "{0}/{1}_{2}_task-faces_posL_neutL_amgyavg.txt".format(output_path,bids_id,ses_id)
        f = open(output_text,"r") 
        z = f.readline()
        f.close()
        all_subject_averages_posL_neuL[s,ses] = float(z[:-1])

        output_text = "{0}/{1}_{2}_task-faces_posF_neutF_amgyavg.txt".format(output_path,bids_id,ses_id)
        f = open(output_text,"r") 
        z = f.readline()
        f.close()
        all_subject_averages_posF_neuF[s,ses] = float(z[:-1])


stat = all_subject_averages_negL_negF
fig = plotPosterStyle(stat,allsubjects)
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

# check that this is right - MDD > CTRL in this case
stat = all_subject_averages_posF_neuF
fig = plotPosterStyle(stat,allsubjects)
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.03)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(stat),0.03)
plt.ylabel('fear > neutral')
plt.xticks(np.arange(2),('Pre NF', 'Post NF'))
#plt.title('A''ignore_neutralF - A''ignore_sadF')
plt.show()


f,p= scipy.stats.ttest_ind(time_diff[MDD_ind],time_diff[HC_ind])
finalp = p/2 # for one-sided t-test for MDD to go down more than HC
print('f is %2.2f' % f)
print('one sided t-test is %2.2f'  % finalp)
