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


results = '/data/jux/cnds/amennen/rtAttenPenn/gazedata/maintenance_ratios.mat'
subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101,102,103,104,105, 106,107,108,109,110,111])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
d = scipy.io.loadmat(results)
d = scipy.io.loadmat(results,struct_as_record=False)
ratios = d['all_gaze_ratios']
ndays = np.shape(ratios)[3]
nemotions = np.shape(ratios)[2]
ntrials = np.shape(ratios)[1]
nsubjects = np.shape(ratios)[0]
# shape of ratios: (7, 12, 4, 3)
# n subjects x 12 trials x 4 emotions x 3 days

# now plot all averages
DYSPHORIC = 1;
THREAT = 2;
NEUTRAL = 3;
POSITIVE = 4;
emotions = ['DYSPHORIC', 'THREAT', 'NEUTRAL', 'POSITIVE']
all_subject_averages = np.nanmean(ratios,axis=1)
nsubs = len(subjects)
alpha=0.5
lw=2
colors=['k', 'r']
nVisits = 4
for em in np.arange(nemotions):
	fig = plt.figure(figsize=(10,7))
	# plot for each subject
	for s in np.arange(nsubs):
		if subjects[s] < 100:
			style = 0
			plt.plot(np.arange(nVisits),all_subject_averages[s,em,:],marker='.', ms=20,color=colors[style],alpha=0.5)
		else:
			style = 1
			plt.plot(np.arange(nVisits),all_subject_averages[s,em,:], marker='.',ms=20,color=colors[style],alpha=0.5)
	plt.errorbar(np.arange(nVisits),np.nanmean(all_subject_averages[HC_ind,em,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(all_subject_averages[HC_ind,em,:],axis=0,nan_policy='omit'), label='HC')
	plt.errorbar(np.arange(nVisits),np.nanmean(all_subject_averages[MDD_ind,em,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(all_subject_averages[MDD_ind,em,:],axis=0,nan_policy='omit'), label='MDD')
	plt.xticks(np.arange(nVisits),('Pre NF', 'Mid NF', 'Post NF', '1M FU'))
	plt.xlabel('Visit')
	plt.ylabel('Time spent')
	plt.ylim([0,0.6])
	plt.title(emotions[em])
	plt.legend()
	plt.show()

