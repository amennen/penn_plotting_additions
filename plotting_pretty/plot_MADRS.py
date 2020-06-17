# purpose: plot MADRS scores by group

import csv
import numpy as np
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
        'size': 22,}

plt.rc('font', **font)
#plt.rc('text', usetex=True)
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, f_classif, GenericUnivariateSelect, SelectKBest, chi2
from sklearn.feature_selection import RFE
import os
#import seaborn as sns
import pandas as pd


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
	        	subject = row[0]
	        	goodrow=row
	        	subject_scores = np.zeros((nVisits,))
	        	subject_scores.fill(np.nan)
	        	nInfo = len(goodrow) 
	        	for v in np.arange(2,nInfo):
	        		if len(goodrow[v]) > 0:
	        			subject_scores[v-2] = np.int(goodrow[v])
	        	MADRS_SCORES[subject] = subject_scores


nsubs = len(MADRS_SCORES)
snumber = [1,2,101,102,103,105,105,106,3,107,4,108,5,6,109,7,110,8,9,10,11,111,112,113]
HC_ind = np.argwhere(np.array(snumber) <100)[:,0]
MDD_ind = np.argwhere(np.array(snumber) > 100)[:,0]
snames = [*MADRS_SCORES]
# now go through each subject and plot visits MADRS
colors = ['k', 'r'] # HC, MDD


fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
	print(s)
	if snumber[s] < 100:
		style = 0
		plt.plot(np.arange(nVisits),MADRS_SCORES[snames[s]],'-',ms=10,color=colors[style],alpha=0.3,lw=2)
	else:
		style = 1
		plt.plot(np.arange(nVisits),MADRS_SCORES[snames[s]],'-',ms=10,color=colors[style],alpha=0.3,lw=2)
HC_scores = [MADRS_SCORES[snames[i]] for i in HC_ind]
MDD_scores = [MADRS_SCORES[snames[i]] for i in MDD_ind]
plt.errorbar(np.arange(nVisits),np.nanmean(HC_scores,axis=0),lw = 2,color=colors[0],yerr=scipy.stats.sem(HC_scores,axis=0,nan_policy='omit'), label='HC',fmt='-o',ms=10)
plt.errorbar(np.arange(nVisits),np.nanmean(MDD_scores,axis=0),lw = 2,color=colors[1],yerr=scipy.stats.sem(MDD_scores,axis=0,nan_policy='omit'), label='MDD',fmt='-o',ms=10)
plt.xticks(np.arange(nVisits),('Pre NF', 'Post NF', '1M FU', '3M FU'))
plt.xlabel('Visit')
plt.ylabel('Depression severity')
#plt.ylabel('MADRS score')
plt.ylim([-1,50])
plt.title('Depression severity over time')
plt.legend(loc=2)
plt.show()

