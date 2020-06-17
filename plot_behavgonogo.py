"""
Functions to help process real-time fMRI data after-the-fact. Processes all the block data from a full run
"""


import numpy as np
import glob 
import sys
import os
import os
import glob
import argparse
import sys
# Add current working dir so main can be run from the top level rtAttenPenn directory
sys.path.append(os.getcwd())
import rtfMRI.utils as utils
import rtfMRI.ValidationUtils as vutils
from rtfMRI.RtfMRIClient import loadConfigFile
from rtfMRI.Errors import ValidationError
from rtAtten.RtAttenModel import getSubjectDayDir
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from rtfMRI.StructDict import StructDict, MatlabStructDict
from sklearn.metrics import roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
#import bokeh
#from bokeh.io import output_notebook, show
#from bokeh.layouts import widgetbox, column, row
#from bokeh.plotting import figure, output_notebook, show
#from bokeh.models import Range1d, Title, Legend
import csv
from anne_additions.aprime_file import aprime,get_blockType ,get_blockData, get_attCategs, get_imCategs, get_trialTypes_and_responses, get_decMatrices_and_aPrimes
#from bokeh.plotting import figure, show, output_file
#from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label



import matplotlib.pyplot as plt
import math

#ATTSCENE_DISTNEUTFACE = 1;
#ATTSCENE_DISTSADFACE = 2;
#ATTNEUTFACE_DISTSCENE = 3;
#ATTHAPPYFACE_DISTSCENE = 4;

# to compare behav bias (aprime sad - aprime neutral) / aprime neutral?

ndays = 3
nruns = 4
subjects = np.array([1,2,3,101,102,103,104,105,106,107])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
nsubs = len(subjects)
all_sadbias = np.zeros((nsubs,ndays))
all_happybias = np.zeros((nsubs,ndays))
realtime = 0
for s in np.arange(nsubs):
	subjectNum = subjects[s]
	for d in np.arange(ndays):
		subjectDay = d+1
		sadbias = np.zeros((nruns))
		happybias = np.zeros((nruns))
		for r in np.arange(4):
			run = r + 1
			data = get_blockData(subjectNum, subjectDay, run)
			run_hitRates, run_missRates, run_FAs, run_CRs, run_aprimes, specificTypes = get_decMatrices_and_aPrimes(data,realtime)
			sad_blocks = np.argwhere(specificTypes == 2)[:,0]
			happy_blocks = np.argwhere(specificTypes == 4)[:,0]
			neut_distface_blocks = np.argwhere(specificTypes == 1)[:,0]
			neut_distscene_blocks = np.argwhere(specificTypes == 3)[:,0]

			aprime_sad = np.nanmean(np.array([run_aprimes[sad_blocks[0]],run_aprimes[sad_blocks[1]]]))
			aprime_distneutface = np.nanmean(np.array([run_aprimes[neut_distface_blocks[0]],run_aprimes[neut_distface_blocks[1]]]))
			aprime_happy = np.nanmean(np.array([run_aprimes[happy_blocks[0]],run_aprimes[happy_blocks[1]]]))
			aprime_distneutscene = np.nanmean(np.array([run_aprimes[neut_distscene_blocks[0]],run_aprimes[neut_distscene_blocks[1]]]))

			# sad bias 
			sadbias[r] = aprime_distneutface - aprime_sad # if sad is distracting would do worse so positive sad bias
			happybias[r] = aprime_distneutscene - aprime_happy # if can't attend to happy would do worse so positive bias

			# specific block is the type of block the person gets
		all_sadbias[s,d] = np.nanmean(sadbias)
		all_happybias[s,d] = np.nanmean(happybias)

linestyles = ['-', ':']
lw = 2
alpha=0.5
ind = np.arange(3) + 1
fig, ax = plt.subplots(figsize=(12,7))
for s in np.arange(nsubs):
	if subjects[s] < 100:
		style = 0
	else:
		style = 1
	plt.plot(ind,all_sadbias[s,:], '-.',linestyle=linestyles[style],linewidth=lw)
plt.bar(ind,np.mean(all_sadbias[HC_ind,:],axis=0),alpha=alpha,label='HC', color='k')
plt.bar(ind,np.mean(all_sadbias[MDD_ind,:],axis=0),alpha=alpha,label='MDD', color='r')
plt.title('Sad bias a prime')
plt.ylabel('A prime neutral - A prime sad')
ax.set_xticks(ind)
ax.set_xticklabels(['Day 1','Day 2',  'Day 3'])
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12,7))
for s in np.arange(nsubs):
	if subjects[s] < 100:
		style = 0
	else:
		style = 1
	plt.plot(ind,all_happybias[s,:], '-.',linestyle=linestyles[style],linewidth=lw)
plt.bar(ind,np.mean(all_happybias[HC_ind,:],axis=0),alpha=alpha,label='HC', color='k')
plt.bar(ind,np.mean(all_happybias[MDD_ind,:],axis=0),alpha=alpha,label='MDD', color='r')
plt.title('Antihappy bias a prime')
plt.ylabel('A prime neutral - A prime happy')
ax.set_xticks(ind)
ax.set_xticklabels(['Day 1','Day 2',  'Day 3'])
plt.legend()
plt.show()





