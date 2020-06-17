
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

powerAtlas = '/data/jux/cnds/amennen/Power/power264MNI_resampled_amygdala.nii.gz'
noise_save_dir = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/resting/clean'
amygdala_mask = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/mni_anat/LAMYG_in_MNI_overlapping.nii.gz'

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


def calculateAmygConnectivity(networkName,correlation_matrix,fullDF,systemDF,all_good_ROI):
	this_ROI = fullDF.ROI[systemDF==networkName].values.astype(int) 
	n_nodes = len(this_ROI)
	# now convert this to the indices
	amyg_indicies_in_matrix = np.where(np.in1d(all_good_ROI,265))[0]
	this_ROI_indices_in_matrix = np.where(np.in1d(all_good_ROI,this_ROI))[0]
	if networkName == 'Sensory/somatomotor Hand':
		# concatenate other one
		other_SMN = fullDF.ROI[systemDF=='Sensory/somatomotor Mouth'].values.astype(int) 
		other_SMN_indices_in_matrix = np.where(np.in1d(all_good_ROI,other_SMN))[0]
		this_ROI_indices_in_matrix = np.concatenate((this_ROI_indices_in_matrix,other_SMN_indices_in_matrix))
	corr_sum=0
	for i in this_ROI_indices_in_matrix:
		this_corr = correlation_matrix[i,amyg_indicies_in_matrix[0]]
		corr_sum = corr_sum + this_corr
	across_ROI_mean = (corr_sum)/n_nodes
	return across_ROI_mean


def calculateWithinConnectivity(networkName,correlation_matrix,fullDF,systemDF,all_good_ROI):
	# find DMN labels
	this_ROI = fullDF.ROI[systemDF==networkName].values.astype(int) 
	# now convert this to the indices
	this_ROI_indices_in_matrix = np.where(np.in1d(all_good_ROI,this_ROI))[0]
	if networkName == 'Sensory/somatomotor Hand':
		# concatenate other one
		other_SMN = fullDF.ROI[systemDF=='Sensory/somatomotor Mouth'].values.astype(int) 
		other_SMN_indices_in_matrix = np.where(np.in1d(all_good_ROI,other_SMN))[0]
		this_ROI_indices_in_matrix = np.concatenate((this_ROI_indices_in_matrix,other_SMN_indices_in_matrix))
	#x,y = np.meshgrid(this_ROI_indices_in_matrix,this_ROI_indices_in_matrix)
	#this_ROI_correlations = correlation_matrix[x,y]
	n_nodes = len(this_ROI)
	corr_sum = 0
	for i in this_ROI_indices_in_matrix:
		for j in this_ROI_indices_in_matrix:
			if i != j:
				this_corr =correlation_matrix[i,j]
				corr_sum = corr_sum + this_corr

	#### CHECK WITH MEICHEN WITH IF THIS IS THE MEAN FIRST OR JUST SUMMING ###
	# within_ROI_sum = np.nansum(this_ROI_correlations)/2 # dividing by 2 because will be double the off-diagonal values
	within_ROI_mean = (corr_sum/2)/np.square(n_nodes)
	#within_ROI_mean = np.nanmean(this_ROI_correlations)/np.square(n_nodes)
	return within_ROI_mean

# for one versus all
# def calculateOneVsAllConnectivity(networkName,correlation_matrix,fullDF,systemDF,all_good_ROI):
# 	this_ROI = fullDF.ROI[systemDF==networkName].values.astype(int) 
# 	this_ROI_indices_in_matrix = np.where(np.in1d(all_good_ROI,this_ROI))[0]
# 	if networkName == 'Sensory/somatomotor Hand':
# 		# concatenate other one
# 		other_SMN = fullDF.ROI[systemDF=='Sensory/somatomotor Mouth'].values.astype(int) 
# 		other_SMN_indices_in_matrix = np.where(np.in1d(all_good_ROI,other_SMN))[0]
# 		this_ROI_indices_in_matrix = np.concatenate((this_ROI_indices_in_matrix,other_SMN_indices_in_matrix))
# 	all_other_indices_in_matrix = [x for x in np.arange(len(all_good_ROI)) if x not in this_ROI_indices_in_matrix]
# 	#x,y = np.meshgrid(this_ROI_indices_in_matrix,all_other_indices_in_matrix)
# 	# this time we're not dividing by 2 because all x values are ROI 1 and all y values are ROI 2
# 	#across_ROI_correlations = correlation_matrix[x,y]
# 	n_nodes_this_network = len(this_ROI)
# 	n_nodes_all_others = len(all_other_indices_in_matrix)
# 	for i in np.arange()
# 	across_ROI_sum = np.nansum(across_ROI_correlations)
# 	across_ROI_mean = across_ROI_sum/(n_nodes_this_network*n_nodes_all_others)
# 	return across_ROI_mean

def calculatePairwiseConnectivity(networkA,networkB,correlation_matrix,fullDF,systemDF,all_good_ROI):
	A_ROI = fullDF.ROI[systemDF==networkA].values.astype(int) 
	A_ROI_indices_in_matrix = np.where(np.in1d(all_good_ROI,A_ROI))[0]
	if networkA == 'Sensory/somatomotor Hand':
		# concatenate other one
		other_SMN = fullDF.ROI[systemDF=='Sensory/somatomotor Mouth'].values.astype(int) 
		other_SMN_indices_in_matrix = np.where(np.in1d(all_good_ROI,other_SMN))[0]
		A_ROI_indices_in_matrix = np.concatenate((A_ROI_indices_in_matrix,other_SMN_indices_in_matrix))
	B_ROI = fullDF.ROI[systemDF==networkB].values.astype(int) 
	B_ROI_indices_in_matrix = np.where(np.in1d(all_good_ROI,B_ROI))[0]
	if networkB == 'Sensory/somatomotor Hand':
		# concatenate other one
		other_SMN = fullDF.ROI[systemDF=='Sensory/somatomotor Mouth'].values.astype(int) 
		other_SMN_indices_in_matrix = np.where(np.in1d(all_good_ROI,other_SMN))[0]
		B_ROI_indices_in_matrix = np.concatenate((B_ROI_indices_in_matrix,other_SMN_indices_in_matrix))
	x,y = np.meshgrid(A_ROI_indices_in_matrix,B_ROI_indices_in_matrix)
	# not dividing by 2 again because again ROI 1 is x and ROI 2 is y so we're not double counting anything
	across_ROI_correlations = correlation_matrix[x,y]
	n_nodes_A = len(A_ROI)
	n_nodes_B = len(B_ROI)
	across_ROI_sum = np.nansum(across_ROI_correlations)
	across_ROI_mean = across_ROI_sum/(n_nodes_A*n_nodes_B)
	# try another way
	corr_sum=0
	for i in A_ROI_indices_in_matrix:
		for j in B_ROI_indices_in_matrix:
			#print(i,j)
			this_corr = correlation_matrix[i,j]
			corr_sum = corr_sum + this_corr
	across_ROI_mean = this_corr/(n_nodes_A*n_nodes_B)
	return across_ROI_mean

# put in check wher eif the std of any voxels in ROI = 0, then skip that vox
nROI = 264
labelsFile = '/data/jag/cnds/amennen/Power/Neuron_consensus_264.csv'
z = pd.read_csv(labelsFile)
complete_labels=z[1:]
ROI = complete_labels['ROI']
system = complete_labels['Suggested System']
all_systems = np.unique(system)
systems_to_keep = ['Default mode','Fronto-parietal Task Control', 
				 'Visual','Subcortical', 'Cingulo-opercular Task Control',  'Salience', 'Ventral attention','Dorsal attention',
				 'Auditory','Sensory/somatomotor Hand', 'Sensory/somatomotor Mouth']
# combine the two sennsory/somatomotor 
n_systems = len(systems_to_keep) - 1
# here we get the ROIs that have each of the labels we don't want
# then we subtract 1 to go to python indices
systems_to_remove = ['Uncertain', 'Cerebellar', 'Memory retrieval?']
systems_to_keep_abbrv = ['DMN', 'FPN', 'VIS', 'SUB', 'CON', 'SAN', 'VAN', 'DAN', 'AUD','SMN']

all_cer_labels = complete_labels.ROI[system=='Cerebellar'].values.astype(int) - 1
all_mem_labels = complete_labels.ROI[system=='Memory retrieval?'].values.astype(int) - 1
all_uncertain_labels = complete_labels.ROI[system=='Uncertain'].values.astype(int) - 1 # go from label to python index
all_bad_labels = np.concatenate((all_cer_labels,all_mem_labels,all_uncertain_labels),axis=0)
# left with 227 regions like beginning of Meichen's (removed the rest for bad signals)
all_network_ind = np.arange(nROI)
all_good_labels = [x for x in all_network_ind if x not in all_bad_labels]
all_good_labels_amyg = all_good_labels + [264] # last index is amygdala
all_good_ROI = np.array(all_good_labels_amyg) + 1 # puts as ROI labels so we can find the specific regions we want
all_FPN_labels = complete_labels.ROI[system=='Fronto-parietal Task Control'].values.astype(int) - 1
nROI_good = len(all_good_labels)

subjects = np.array([3,4,5,6,7,8,9,10,11,106,107,108,109,110,111,112,113,114,115])
nSub = len(subjects)
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
sessions = [1,3]
nDays = len(sessions)
average_within_mat = np.zeros((n_systems,n_systems,nSub,nDays))
average_one_vs_all = np.zeros((n_systems,nSub,nDays))
amyg_con = np.zeros((nSub,nDays))
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
		masker = NiftiLabelsMasker(labels_img=powerAtlas, standardize=True,
		                           memory='nilearn_cache', verbose=5)
		time_series = masker.fit_transform(cleaned_image) # now data is n time points x 264 nodes
		time_series_good_labels = time_series[:,all_good_labels_amyg] # now data is in n time points x 227 nodes
		time_series_df = pd.DataFrame(time_series_good_labels)
		correlation_matrix = np.array(time_series_df.corr(method='pearson'))
		#correlation_measure = ConnectivityMeasure(kind='correlation')
		#correlation_matrix = correlation_measure.fit_transform([time_series_good_labels])[0] # takes correlation for all 227 nodes
		np.fill_diagonal(correlation_matrix,np.nan) # to make sure you don't get the same node in the within connectivity difference
		for row in np.arange(n_systems):
			for col in np.arange(n_systems):
				if row == col: # diagonal
					average_within_mat[row,col,s,ses] = calculateWithinConnectivity(systems_to_keep[row],correlation_matrix,complete_labels,system,all_good_ROI)
				else:
					average_within_mat[row,col,s,ses] = calculatePairwiseConnectivity(systems_to_keep[row],systems_to_keep[col],correlation_matrix,complete_labels,system,all_good_ROI)
			# now calculate oneVsAll
			#average_one_vs_all[row,s,ses] = calculateOneVsAllConnectivity(systems_to_keep[row],correlation_matrix,complete_labels,system,all_good_ROI)
		amyg_con[s,ses] = calculateAmygConnectivity(systems_to_keep[1],correlation_matrix,complete_labels,system,all_good_ROI)
# first plot DMN connectivity
dmn_connectivity = average_within_mat[0,0,:,:] # subjects x visit
stat = dmn_connectivity
fig = plotPosterStyle(stat,subjects)
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.05)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(stat),0.05)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat(p/2,0,1,np.nanmax(stat),0.1)
plt.ylabel('within DMN connectivity')
plt.xticks(np.arange(2))
plt.show()


stat = amyg_con
fig = plotPosterStyle(stat,subjects)
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.05)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(stat),0.05)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat(p/2,0,1,np.nanmax(stat),0.1)
plt.ylabel('FPN -> LA connectivity')
plt.xticks(np.arange(2))
plt.show()


# san_connectivity = average_within_mat[5,5,:,:] # subjects x visit
# stat = san_connectivity
# fig = plotPosterStyle(stat,subjects)
# x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
# t,p = scipy.stats.ttest_ind(x,y)
# addSingleStat(p/2,0,np.nanmax(stat),0.05)
# x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
# t,p = scipy.stats.ttest_ind(x,y)
# addSingleStat(p/2,1,np.nanmax(stat),0.05)
# x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,1])
# t,p = scipy.stats.ttest_ind(x,y)
# addComparisonStat(p/2,0,1,np.nanmax(stat),0.1)
# plt.ylabel('within DMN connectivity')
# plt.xticks(np.arange(2))
# plt.show()


fpn_connectivity = average_within_mat[1,1,:,:] # subjects x visit
stat = fpn_connectivity
fig = plotPosterStyle(stat,subjects)
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.05)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(stat),0.05)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat(p/2,0,1,np.nanmax(stat),0.1)
plt.ylabel('within FPN connectivity')
plt.xticks(np.arange(2))
plt.show()

dmn_to_fpn = average_within_mat[1,0,:,:]
stat = dmn_to_fpn
fig = plotPosterStyle(stat,subjects)
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.0005)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(stat),0.0005)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat(p/2,0,1,np.nanmax(stat),0.001)
plt.ylabel('within FPN connectivity')
plt.xticks(np.arange(2))
plt.show()



con_connectivity = average_within_mat[4,4,:,:]
stat = con_connectivity
fig = plotPosterStyle(stat,subjects)
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.05)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(stat),0.05)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat(p/2,0,1,np.nanmax(stat),0.1)
plt.ylabel('within CON connectivity')
plt.xticks(np.arange(2))
plt.show()

# maybe all negative - do task positive vs. task negative
# task positive: DAN, CON, FPN
# task negative: DMN, SAN
# sensory networks: SMN VIS
task_positive_networks = np.array([1,4,7])
task_negative_networks = np.array([0,5])
sensory_networks = np.array([2,9])
tp_diag = np.zeros((nSub,nDays))
tn_diag = np.zeros((nSub,nDays))
s_diag = np.zeros((nSub,nDays))
for s in np.arange(nSub):
	for d in np.arange(nDays):
		tp_diag[s,d] = np.mean(np.diag(average_within_mat[task_positive_networks,task_positive_networks,s,d]))
		tn_diag[s,d] = np.mean(np.diag(average_within_mat[task_negative_networks,task_negative_networks,s,d]))
		s_diag[s,d] = np.mean(np.diag(average_within_mat[sensory_networks,sensory_networks,s,d]))

stat=tp_diag
fig,ax = plotPosterStyle_DF(stat,subjects)
plt.title('task positive networks')
plt.show()

stat=tn_diag
fig,ax = plotPosterStyle_DF(stat,subjects)
plt.title('task negative networks')
plt.show()

stat=s_diag
fig,ax = plotPosterStyle_DF(stat,subjects)
plt.title('sensory networks')
plt.show()

# look at DMN - FPN connection # TO DO: pairwise!
dmn_fpn = average_within_mat[0,1,:,:]
stat = dmn_fpn
fig = plotPosterStyle(stat,subjects)
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.05)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(stat),0.05)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat(p/2,0,1,np.nanmax(stat),0.1)
plt.ylabel('within FPN connectivity')
plt.xticks(np.arange(2))
plt.show()

# calculate change in DMN connectivity
system = 0 # 0 = DMN, 1 = FPN
row=system
col=row
linestyles = ['-', ':']
colors=['k', 'r']
nVisits = 2


system = 0 # 0 = DMN, 1 = FPN
row=system
col=row
linestyles = ['-', ':']
colors=['k', 'r']
nVisits = 2


nSystems = 10
within_matrix = np.zeros((nSub*2*nSystems,1))
for sub in np.arange(nSub):
	for sys in np.arange(nSystems):
		within_matrix[sys+(nSystems*2*sub),0] = average_within_mat[sys,sys,sub,0]
		print(sys+(nSystems*2*sub))
		within_matrix[sys+(nSystems*2*sub)+nSystems,0] = average_within_mat[sys,sys,sub,1]
		print(sys+(nSystems*2*sub)+nSystems)
ses_vec = np.ones((nSystems*2,1))
ses_vec[nSystems:,0] = 3
ses_vec = np.tile(ses_vec,(nSub,1))
group_vec = np.zeros((nSub,1))
group_vec[MDD_ind,0] = 1
group_vec = np.reshape(np.repeat(group_vec,(nSystems*2)),(nSub*2*nSystems,1))
subject_vec = np.reshape(np.repeat(subjects,nSystems*2),(nSub*2*nSystems,1))
network_vec = np.reshape(np.tile(np.arange(nSystems),2*nSub),(nSub*2*nSystems,1))
# make a dataframe -- starting from array
# want: subjectNumber, group, visit, 
all_data = np.concatenate((subject_vec,group_vec,ses_vec,within_matrix,network_vec),axis=1)
all_cols = ['subject', 'group', 'session'] + systems_to_keep_abbrv 
data = pd.DataFrame(data=all_data,columns=['subject', 'group', 'session', 'CON', 'network']  )

fig = plt.figure()
g = sns.catplot(data=data,x='network',y='CON', hue='group', kind='bar',col='session',ci=68,palette=['k', 'r'],alpha=0.5)
g.set_xticklabels(systems_to_keep,rotation=45,fontsize=10)
g.set_ylabels('within network connectivity')
# now go through each option and test if significantly different
for ax,ses in zip(g.axes.flat,np.arange(nDays)):
	for sys in np.arange(nSystems):
		this_within_system = average_within_mat[sys,sys,:,ses]
		x,y = nonNan(this_within_system[HC_ind],this_within_system[MDD_ind])
		t,p = scipy.stats.ttest_ind(x,y)
		addSingleStat_ax(p,sys,np.nanmax(stat),0.03,ax)

plt.show()

dmn_connectivity = average_within_mat[0,0,:,:] # subjects x visit
dmn_connectivity_change = dmn_connectivity[:,1] - dmn_connectivity[:,0]
M = getMADRSscoresALL()

# look at each day
fig = plt.figure(figsize=(10,7))
ax1 = plt.subplot(1,2,1)
for s in np.arange(nSub-1):
	subjectNum  = subjects[s]
	this_sub_madrs = M[subjectNum]
	if subjectNum < 100:
		style = 0
	elif subjectNum > 100:
		style = 1
	plt.plot(dmn_connectivity[s,0],this_sub_madrs[0],marker='.',ms=20,color=colors[style],alpha=0.5)
plt.xlim([0,.17])
plt.ylim([-1,30])
ax2 = plt.subplot(1,2,2)
for s in np.arange(nSub-1):
	subjectNum  = subjects[s]
	this_sub_madrs = M[subjectNum]
	if subjectNum < 100:
		style = 0
	elif subjectNum > 100:
		style = 1
	plt.plot(dmn_connectivity[s,1],this_sub_madrs[1],marker='.',ms=20,color=colors[style],alpha=0.5)
plt.xlim([0,.17])
plt.ylim([-1,30])
plt.xlabel('DMN Connectivity Change 3 - 1')
plt.ylabel('MADRS Change 3 - 1')
plt.show()


# transFigure = fig.transFigure.inverted()
# for i in range(nSub):
# 	subjectNum = subjects[i]
# 	x_y_1 = np.array([dmn_connectivity[s,0],M[subjectNum][0]])
# 	xy1 = transFigure.transform(ax1.transData.transform([x_y_1[0],x_y_1[1]]))
# 	x_y_2 = np.array([dmn_connectivity[s,1],M[subjectNum][1]])
# 	xy2 = transFigure.transform(ax2.transData.transform([x_y_2[0],x_y_2[1]]))
# 	line = matplotlib.lines.Line2D((xy1[0],xy2[0]),(xy1[1],xy2[1]),
#                                    transform=fig.transFigure)
# 	fig.lines.append(line)


dmn_connectivity2 = average_one_vs_all[system,:,:] # subjects x visit
dmn_connectivity_change2 = dmn_connectivity2[:,1] - dmn_connectivity2[:,0]
# now get all MADRS score changes for that subject
fig = plt.figure(figsize=(10,7))
for s in np.arange(nSub-1):
	subjectNum  = subjects[s]
	this_sub_madrs = M[subjectNum]
	madrs_change = this_sub_madrs[1] - this_sub_madrs[0]
	if subjectNum < 100:
		style = 0
	elif subjectNum > 100:
		style = 1
	plt.plot(dmn_connectivity_change[s],madrs_change,marker='.',ms=20,color=colors[style],alpha=0.5)
plt.xlabel('DMN Connectivity Change 3 - 1')
plt.ylabel('MADRS Change 3 - 1')
plt.show()
scipy.stats.


fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nSub):
	if subjects[s] < 100:
		style = 0
		plt.plot(np.arange(nVisits),average_within_mat[row,col,s,:],marker='.', ms=20,color=colors[style],alpha=0.5)
	else:
		style = 1
		plt.plot(np.arange(nVisits),average_within_mat[row,col,s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(average_within_mat[row,col,HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(average_within_mat[row,col,HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(average_within_mat[row,col,MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(average_within_mat[row,col,MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Post NF'))
plt.xlabel('Visit')
plt.title('Row %i Col %i' % (row,col))
plt.title('%s Within-Network Connectivity'% systems_to_keep_abbrv[system])
plt.legend()
plt.show()
# now test significance
print('FIRST DAY')
print(scipy.stats.ttest_ind(average_within_mat[row,col,HC_ind,0],average_within_mat[row,col,MDD_ind,0]))
print('LAST DAY')
print(scipy.stats.ttest_ind(average_within_mat[row,col,HC_ind,1],average_within_mat[row,col,MDD_ind,1]))