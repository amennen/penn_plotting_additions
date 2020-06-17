
import glob
import pandas as pd
import numpy as np
from subprocess import call
import time
import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s - %(message)s')
import numpy as np
import pickle
import nibabel as nib
import nilearn
from nilearn.image import resample_to_img
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.plotting import show
from nilearn.plotting import plot_roi
from nilearn import image
from nilearn.masking import apply_mask
# get_ipython().magic('matplotlib inline')
import scipy
import matplotlib
import matplotlib.pyplot as plt
from nilearn import image
from nilearn.input_data import NiftiMasker
#from nilearn import plotting
from nilearn.masking import apply_mask
from nilearn.image import load_img
from nilearn.image import new_img_like
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
import csv
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
import seaborn as sns
import pandas as pd
import csv
from scipy import stats
import sys
from sklearn.utils import shuffle
import random
from datetime import datetime
random.seed(datetime.now())
from nilearn.image import new_img_like
import scipy.stats as sstats  # type: ignore
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from anne_additions.plotting_pretty.commonPlotting import *
from nilearn import masking
powerAtlas = '/data/jag/cnds/amennen/Power/power264MNI_resampled.nii.gz'
noise_save_dir = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/resting/clean'
ROI_DIR = '/data/jux/cnds/amennen/rtAttenPenn/MNI_things/clusters'
ACC_ROI = "{0}/cluster{1}sphere.nii.gz".format(ROI_DIR,1)  # Anterior cingulate is cluster # 1
amygdala_mask = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/mni_anat/LAMYG_in_MNI_overlapping.nii.gz'
MNI_dir = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/mni_anat/'
dlPFC_neurosynth = MNI_dir + 'dlpfc_association-test_z_FDR_0.01_thr_5.5_resampled.nii.gz'
# to do: combine amygdala mask with the ACC ROI so they have different labels
dorsal_acc = "{0}/cluster{1}sphere.nii.gz".format(ROI_DIR,0+1)
# cmd = "fslmaths {0} -mul 2 {1}/LAMYG_in_MNI_overlapping_mul2.nii.gz".format(amygdala_mask,MNI_dir)
# call(cmd,shell=True)
# cmd = "fslmaths {0}/LAMYG_in_MNI_overlapping_mul2.nii.gz -add {1} {0}/LAMYG_ACC.nii.gz".format(MNI_dir,ACC_ROI)
# cmd = "fslmaths {0}/LAMYG_in_MNI_overlapping_mul2.nii.gz -add {1} {0}/LAMYG_ACC.nii.gz".format(MNI_dir,ACC_ROI)
cmd = "fslmaths {0}/LAMYG_in_MNI_overlapping_mul2.nii.gz -add {1} {0}/LAMYG_dlPFC.nii.gz".format(MNI_dir,dlPFC_neurosynth)
cmd = "fslmaths {0}/LAMYG_in_MNI_overlapping_mul2.nii.gz -add {1} {0}/LAMYG_dorsal_acc.nii.gz".format(MNI_dir,dorsal_acc)

# call(cmd,shell=True)
combined_mask = "{0}/LAMYG_ACC.nii.gz".format(MNI_dir)
combined_mask2 = "{0}/LAMYG_dlPFC.nii.gz".format(MNI_dir)

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



subjects = np.array([3,4,5,6,7,8,9,10,11,106,107,108,109,110,111,112,113,114])
nSub = len(subjects)
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
sessions = [1,3]
nDays = len(sessions)
cross_corr = np.zeros((nSub,nDays))
# NOW CALCULATE DATA FOR SUBJECTS
for s in np.arange(nSub):
	subjectNum=subjects[s]
	bids_id = 'sub-{0:03d}'.format(subjectNum)
	for ses in np.arange(nDays):
		subjectDay=sessions[ses]
		ses_id = 'ses-{0:02d}'.format(subjectDay)
		clean_path = noise_save_dir + '/' + bids_id + '/' + ses_id
		cleaned_image = '{0}/{1}_{2}_task_rest_glm.nii.gz'.format(clean_path,bids_id,ses_id)

		#cleaned_image_data = nib.load(cleaned_image).get_fdata()
		# doing standardize = True here at least makes it so voxels outside of brain would have 0 std and not be included
		masker = NiftiLabelsMasker(labels_img=combined_mask, standardize=True,smoothing_fwhm=5,
		                           memory='nilearn_cache', verbose=5)
		time_series = masker.fit_transform(cleaned_image) # now data is n tim

		# correlation_measure = ConnectivityMeasure(kind='correlation')
		# #correlation_matrix = correlation_measure.fit_transform([time_series])[0]
		# cross_corr[s,ses]  = correlation_matrix[0,1]
		cross_corr[s,ses]  =scipy.stats.pearsonr(time_series[:,0],time_series[:,1])[0]

fig,ax = plotPosterStyle_DF(cross_corr,subjects)
#plt.ylim([0,1])
plt.xticks(np.arange(2),('NF 1', 'NF 3'))
x,y = nonNan(cross_corr[HC_ind,0],cross_corr[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.1,.1,0.4,0.05,0,'$MDD < HC$')
#addSingleStat(p/2,0,np.nanmax(cross_corr),0.01)
plt.ylabel('Func connectivity: LA - dorsal ACC')
# x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,2])
# t,p = scipy.stats.ttest_ind(x,y)
# addSingleStat(p/2,2,np.nanmax(stat),0.01)
# x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
# t,p = scipy.stats.ttest_rel(x,y)
# addComparisonStat(p/2,0,2,np.nanmax(stat),0.05)
plt.show()


