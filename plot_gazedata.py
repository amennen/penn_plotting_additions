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


results = '/data/jag/cnds/amennen/rtAttenPenn/gazedata/gaze_ratios.mat'
subjects = np.array([1,2,3,101,102,103,104,105, 106,107])
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

alpha=0.5
lw=2
ind = np.arange(ndays)
linestyles = ['-', ':']
for em in np.arange(nemotions):
	fig, ax = plt.subplots(figsize=(12,7))
	for s in np.arange(nsubjects):
		if subjects[s] < 100:
			style = 0
		else:
			style = 1
		plt.plot(ind,all_subject_averages[s,em,:], '-.',linestyle=linestyles[style],linewidth=lw)
	plt.bar(ind,np.nanmean(all_subject_averages[HC_ind,em,:],axis=0),alpha=alpha,label='HC', color='k')
	plt.bar(ind,np.nanmean(all_subject_averages[MDD_ind,em,:],axis=0),alpha=alpha,label='MDD', color='r')
	plt.title(emotions[em])
	plt.ylabel('Ratio time spent')
	ax.set_xticks(ind)
	ax.set_xticklabels(['Day 1', 'Day 2', 'Day 3'])
	plt.legend()
	plt.show()

