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
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, f_classif, GenericUnivariateSelect, SelectKBest, chi2
from sklearn.feature_selection import RFE
import os
#import seaborn as sns
import pandas as pd



FMRIPREP_DIR='/data/jag/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/fmriprep/'
FSL_DIR='/data/jag/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/fsl/first_level/'
mask = 'AMYG.nii.gz'
Lmask = 'LAMYG.nii.gz'
Rmask = 'RAMYG.nii.gz'
MFG = 'MFG.nii.gz'
LMFG = 'LMFG.nii.gz'
RMFG = 'RMFG.nii.gz'
subjects = np.array([1,2,3,4,5,101,102,103,104, 105, 106, 107,108])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
nsubs = len(subjects)
all_subject_averages_fear_neutral = np.zeros((nsubs,2))
all_subject_averages_happy_neutral = np.zeros((nsubs,2))
L_all_subject_averages_fear_neutral = np.zeros((nsubs,2))
R_all_subject_averages_fear_neutral = np.zeros((nsubs,2))
L_all_subject_averages_happy_neutral = np.zeros((nsubs,2))
R_all_subject_averages_happy_neutral = np.zeros((nsubs,2))
MFG_all_subject_averages_fear_neutral = np.zeros((nsubs,2))
MFG_all_subject_averages_happy_neutral = np.zeros((nsubs,2))
LMFG_all_subject_averages_fear_neutral = np.zeros((nsubs,2))
RMFG_all_subject_averages_fear_neutral = np.zeros((nsubs,2))
LMFG_all_subject_averages_happy_neutral = np.zeros((nsubs,2))
RMFG_all_subject_averages_happy_neutral = np.zeros((nsubs,2))

days = np.array([1,3])
for s in np.arange(nsubs):
	for d in np.arange(len(days)):
		subjectNum = subjects[s]
		subjectDay = days[d]
		bids_id = 'sub-{0:03d}'.format(subjectNum)
		ses_id = 'ses-{0:02d}'.format(subjectDay)
		func_day_path=os.path.join(FMRIPREP_DIR,bids_id,ses_id, 'func')
		firstlevel_path = os.path.join(FSL_DIR,bids_id,ses_id)
		mask_file = os.path.join(func_day_path, mask)
		L_mask_file = os.path.join(func_day_path,Lmask)
		R_mask_file = os.path.join(func_day_path,Rmask)
		maskOBJ = nibabel.load(mask_file)	
		mask_data = maskOBJ.get_data().astype(int)
		logical_mask = mask_data > 0
		# no do the same for left mask
		LmaskOBJ = nibabel.load(L_mask_file)
		Lmask_data = LmaskOBJ.get_data().astype(int)
		Llogical_mask = Lmask_data > 0
		# no do the same for right mask
		RmaskOBJ = nibabel.load(R_mask_file)
		Rmask_data = RmaskOBJ.get_data().astype(int)
		Rlogical_mask = Rmask_data > 0

		# now get all MFG masks ####
		MFG_mask_file = os.path.join(func_day_path,MFG)
		MFG_mask_data = nibabel.load(MFG_mask_file).get_data().astype(int)
		MFG_logical = MFG_mask_data > 0
		LMFG_mask_file = os.path.join(func_day_path,LMFG)
		LMFG_mask_data = nibabel.load(LMFG_mask_file).get_data().astype(int)
		LMFG_logical = LMFG_mask_data > 0
		RMFG_mask_file = os.path.join(func_day_path, RMFG)
		RMFG_mask_data = nibabel.load(RMFG_mask_file).get_data().astype(int)
		RMFG_logical = RMFG_mask_data > 0

		
		faces1_file = os.path.join(firstlevel_path,'faces1_T1w_final.feat','stats', 'cope5.nii.gz')
		faces2_file = os.path.join(firstlevel_path,'faces2_T1w_final.feat','stats','cope5.nii.gz')
		# go to patterns folder to see how he made the labels
		# now get the top 1000 voxels for scene
		ftemp = nibabel.load(faces1_file).get_data()
		faces1_stat = np.mean(ftemp[logical_mask])
		Lfaces1_stat = np.mean(ftemp[Llogical_mask])
		Rfaces1_stat = np.mean(ftemp[Rlogical_mask])
		MFG_faces1 = np.mean(ftemp[MFG_logical])
		LMFG_faces1 = np.mean(ftemp[LMFG_logical])
		RMFG_faces1 = np.mean(ftemp[RMFG_logical])
		ftemp = nibabel.load(faces2_file).get_data()
		faces2_stat = np.mean(ftemp[logical_mask])
		Lfaces2_stat = np.mean(ftemp[Llogical_mask])
		Rfaces2_stat = np.mean(ftemp[Rlogical_mask])
		MFG_faces2 = np.mean(ftemp[MFG_logical])
		LMFG_faces2 = np.mean(ftemp[LMFG_logical])
		RMFG_faces2 = np.mean(ftemp[RMFG_logical])
		all_subject_averages_fear_neutral[s,d] = np.mean(np.array([faces1_stat,faces2_stat]))
		# now for left mask only
		L_all_subject_averages_fear_neutral[s,d] = np.mean(np.array([Lfaces1_stat,Lfaces2_stat]))
		# now for right mask only
		R_all_subject_averages_fear_neutral[s,d] = np.mean(np.array([Rfaces1_stat,Rfaces2_stat]))
		MFG_all_subject_averages_fear_neutral[s,d] = np.mean(np.array([MFG_faces1,MFG_faces2]))
		LMFG_all_subject_averages_fear_neutral[s,d] = np.mean(np.array([LMFG_faces1,LMFG_faces2]))
		RMFG_all_subject_averages_fear_neutral[s,d] = np.mean(np.array([RMFG_faces1,RMFG_faces2]))
		faces1_file = os.path.join(firstlevel_path,'faces1_T1w_final.feat','stats', 'cope6.nii.gz')
		faces2_file = os.path.join(firstlevel_path,'faces2_T1w_final.feat','stats','cope6.nii.gz')
		# go to patterns folder to see how he made the labels
		# now get the top 1000 voxels for scene
		ftemp = nibabel.load(faces1_file).get_data()
		faces1_stat = np.mean(ftemp[logical_mask])
		Lfaces1_stat = np.mean(ftemp[Llogical_mask])
		Rfaces1_stat = np.mean(ftemp[Rlogical_mask])
		MFG_faces1 = np.mean(ftemp[MFG_logical])
		LMFG_faces1 = np.mean(ftemp[LMFG_logical])
		RMFG_faces1 = np.mean(ftemp[RMFG_logical])
		ftemp = nibabel.load(faces2_file).get_data()
		faces2_stat = np.mean(ftemp[logical_mask])
		Lfaces2_stat = np.mean(ftemp[Llogical_mask])
		Rfaces2_stat = np.mean(ftemp[Rlogical_mask])
		MFG_faces2 = np.mean(ftemp[MFG_logical])
		LMFG_faces2 = np.mean(ftemp[LMFG_logical])
		RMFG_faces2 = np.mean(ftemp[RMFG_logical])
		all_subject_averages_happy_neutral[s,d] = np.mean(np.array([faces1_stat,faces2_stat]))
		# now for left mask only
		L_all_subject_averages_happy_neutral[s,d] = np.mean(np.array([Lfaces1_stat,Lfaces2_stat]))
		# now for right mask only
		R_all_subject_averages_happy_neutral[s,d] = np.mean(np.array([Rfaces1_stat,Rfaces2_stat]))
		
		MFG_all_subject_averages_happy_neutral[s,d] = np.mean(np.array([MFG_faces1,MFG_faces2]))
		LMFG_all_subject_averages_happy_neutral[s,d] = np.mean(np.array([LMFG_faces1,LMFG_faces2]))
		RMFG_all_subject_averages_happy_neutral[s,d] = np.mean(np.array([RMFG_faces1,RMFG_faces2]))


linestyles = ['-', ':']
colors=['k', 'r']
nVisits = 2


fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),all_subject_averages_fear_neutral[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(all_subject_averages_fear_neutral[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(all_subject_averages_fear_neutral[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(all_subject_averages_fear_neutral[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(all_subject_averages_fear_neutral[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Post NF'))
plt.xlabel('Visit')
plt.ylabel('Amygdala activity Fear > Neutral')
plt.ylim([-30,30])
plt.title('Amygdala Fear > Neutral')
plt.legend()
plt.show()

# now do for L/R Separately
fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),L_all_subject_averages_fear_neutral[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(L_all_subject_averages_fear_neutral[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(L_all_subject_averages_fear_neutral[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(L_all_subject_averages_fear_neutral[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(L_all_subject_averages_fear_neutral[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Post NF'))
plt.xlabel('Visit')
plt.ylim([-30,30])
plt.ylabel('Left Amygdala activity Fear > Neutral')
plt.title('Left Amygdala Fear > Neutral')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),R_all_subject_averages_fear_neutral[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(R_all_subject_averages_fear_neutral[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(R_all_subject_averages_fear_neutral[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(R_all_subject_averages_fear_neutral[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(R_all_subject_averages_fear_neutral[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Post NF'))
plt.xlabel('Visit')
plt.ylim([-30,30])
plt.ylabel('Right Amygdala activity Fear > Neutral')
plt.title('Right Amygdala Fear > Neutral')
plt.legend()
plt.show()

# NOW HAPPY > NEUTRAL

fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),all_subject_averages_happy_neutral[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(all_subject_averages_happy_neutral[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(all_subject_averages_happy_neutral[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(all_subject_averages_happy_neutral[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(all_subject_averages_happy_neutral[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Post NF'))
plt.xlabel('Visit')
plt.ylabel('Amygdala activity Happy > Neutral')
plt.ylim([-30,30])
plt.title('Amygdala Happy > Neutral')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),L_all_subject_averages_happy_neutral[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(L_all_subject_averages_happy_neutral[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(L_all_subject_averages_happy_neutral[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(L_all_subject_averages_happy_neutral[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(L_all_subject_averages_happy_neutral[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Post NF'))
plt.xlabel('Visit')
plt.ylabel('Left Amygdala activity Happy > Neutral')
plt.title('Left Amygdala Happy > Neutral')
plt.ylim([-30,30])
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),R_all_subject_averages_happy_neutral[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(R_all_subject_averages_happy_neutral[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(R_all_subject_averages_happy_neutral[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(R_all_subject_averages_happy_neutral[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(R_all_subject_averages_happy_neutral[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Post NF'))
plt.xlabel('Visit')
plt.ylabel('Right Amygdala activity Happy > Neutral')
plt.ylim([-30,30])
plt.title('Right Amygdala Happy > Neutral')
plt.legend()
plt.show()

############## REPEAT PLOTS FOR MFG ##################################################################################################

fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),MFG_all_subject_averages_fear_neutral[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(MFG_all_subject_averages_fear_neutral[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(MFG_all_subject_averages_fear_neutral[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(MFG_all_subject_averages_fear_neutral[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(MFG_all_subject_averages_fear_neutral[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Post NF'))
plt.xlabel('Visit')
plt.ylabel('MFG activity Fear > Neutral')
plt.ylim([-30,30])
plt.title('MFG Fear > Neutral')
plt.legend()
plt.show()

# now do for L/R Separately
fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),LMFG_all_subject_averages_fear_neutral[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(LMFG_all_subject_averages_fear_neutral[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(LMFG_all_subject_averages_fear_neutral[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(LMFG_all_subject_averages_fear_neutral[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(LMFG_all_subject_averages_fear_neutral[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Post NF'))
plt.xlabel('Visit')
plt.ylim([-30,30])
plt.ylabel('Left MFG activity Fear > Neutral')
plt.title('Left MFG Fear > Neutral')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),RMFG_all_subject_averages_fear_neutral[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(RMFG_all_subject_averages_fear_neutral[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(RMFG_all_subject_averages_fear_neutral[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(RMFG_all_subject_averages_fear_neutral[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(RMFG_all_subject_averages_fear_neutral[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Post NF'))
plt.xlabel('Visit')
plt.ylim([-30,30])
plt.ylabel('Right MFG activity Fear > Neutral')
plt.title('Right MFG Fear > Neutral')
plt.legend()
plt.show()

# NOW HAPPY > NEUTRAL

fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),MFG_all_subject_averages_happy_neutral[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(MFG_all_subject_averages_happy_neutral[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(MFG_all_subject_averages_happy_neutral[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(MFG_all_subject_averages_happy_neutral[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(MFG_all_subject_averages_happy_neutral[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Post NF'))
plt.xlabel('Visit')
plt.ylabel('MFG activity Happy > Neutral')
plt.ylim([-30,30])
plt.title('MFG Happy > Neutral')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),LMFG_all_subject_averages_happy_neutral[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(LMFG_all_subject_averages_happy_neutral[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(LMFG_all_subject_averages_happy_neutral[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(LMFG_all_subject_averages_happy_neutral[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(LMFG_all_subject_averages_happy_neutral[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Post NF'))
plt.xlabel('Visit')
plt.ylabel('Left MFG activity Happy > Neutral')
plt.title('Left MFG Happy > Neutral')
plt.ylim([-30,30])
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
        if subjects[s] < 100:
                style = 0
        else:
                style = 1
        plt.plot(np.arange(nVisits),RMFG_all_subject_averages_happy_neutral[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(RMFG_all_subject_averages_happy_neutral[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(RMFG_all_subject_averages_happy_neutral[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(RMFG_all_subject_averages_happy_neutral[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(RMFG_all_subject_averages_happy_neutral[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Post NF'))
plt.xlabel('Visit')
plt.ylabel('Right MFG activity Happy > Neutral')
plt.ylim([-30,30])
plt.title('Right MFG Happy > Neutral')
plt.legend()
plt.show()


