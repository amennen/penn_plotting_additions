"""
Functions to help process real-time fMRI data after-the-fact. Processes all the block data from a full run
"""


import numpy as np
import glob 
import sys
import os
import os
import scipy
import glob
import argparse
import sys
# Add current working dir so main can be run from the top level rtAttenPenn directory
sys.path.append(os.getcwd())
import rtfMRI.utils as utils
import rtfMRI.ValidationUtils as vutils
from rtfMRI.RtfMRIClient import loadConfigFile
from rtfMRI.Errors import ValidationError

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

#ATTSCENE_DISTNEUTFACE = 1;
#ATTSCENE_DISTSADFACE = 2;
#ATTNEUTFACE_DISTSCENE = 3;
#ATTHAPPYFACE_DISTSCENE = 4;

# to compare behav bias (aprime sad - aprime neutral) / aprime neutral?
realtime = 0
ndays = 3
nruns = 4
subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101,102,103,104,105,106,107,108,109,110,111,112,113,114])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
nsubs = len(subjects)
all_sadbias = np.zeros((nsubs,ndays))
all_happybias = np.zeros((nsubs,ndays))
all_neutralface = np.zeros((nsubs,ndays))
all_neutralscene = np.zeros((nsubs,ndays))
fa_neutralface = np.zeros((nsubs,ndays))
fa_neutralscene = np.zeros((nsubs,ndays))
fa_sadface = np.zeros((nsubs,ndays))
fa_happyface = np.zeros((nsubs,ndays))
for s in np.arange(nsubs):
	subjectNum = subjects[s]
	for d in np.arange(ndays):
		print('s,d is %i,%i' % ((subjectNum,d)))
		subjectDay = d+1
		sadbias = np.zeros((nruns))
		happybias = np.zeros((nruns))
		neutralface = np.zeros((nruns))
		neutralscene = np.zeros((nruns))
		for r in np.arange(4):
			run = r + 1
			# subject 108 didn't do 4 runs on day 2
			if subjectNum == 108 and subjectDay == 2 and run == 1:
				sadbias[r] = np.nan
				happybias[r] = np.nan
				neutralface[r] = np.nan
				neutralscene[r] = np.nan
			else:
				data = get_blockData(subjectNum, subjectDay, run)
				if data != -1:
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
					neutralface[r] = aprime_distneutscene
					neutralscene[r] = aprime_distneutface

			# specific block is the type of block the person gets
		all_sadbias[s,d] = np.nanmean(sadbias)
		all_happybias[s,d] = np.nanmean(happybias)
		all_neutralface[s,d] = np.nanmean(neutralface)
		all_neutralscene[s,d] = np.nanmean(neutralscene)

# get bias difference by day
all_sadbias_diff = all_sadbias[:,2] - all_sadbias[:,0]
M = getMADRSscoresALL()
d1,d2,d3 = getMADRSdiff(M,subjects)

colors = ['k', 'r'] # HC, MDD
fig = plt.figure(figsize=(10,7))
for s in np.arange(nsubs):
  subjectNum  = subjects[s]
  this_sub_madrs = M[subjectNum]
  madrs_change = this_sub_madrs[1] - this_sub_madrs[0]
  print(madrs_change)
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(all_sadbias_diff[s],madrs_change,marker='.',ms=20,color=colors[style],alpha=0.5)
plt.xlabel('Sadbias diff 5 - 1')
plt.ylabel('MADRS Change 3 - 1')
plt.show()


linestyles = ['-', ':']
colors=['k', 'r']
nVisits = ndays

fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
	if subjects[s] < 100:
		style = 0
		plt.plot(np.arange(nVisits),all_sadbias[s,:],marker='.', ms=20,color=colors[style],alpha=0.5)
	else:
		style = 1
		plt.plot(np.arange(nVisits),all_sadbias[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(all_sadbias[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(all_sadbias[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(all_sadbias[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(all_sadbias[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Mid NF', 'Post NF', '1M FU'))
plt.xlabel('Visit')
plt.ylabel('A` neutral - A` sad')
plt.ylim([-.3,.3])
plt.title('Negative = better with sad')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
	if subjects[s] < 100:
		style = 0
		plt.plot(np.arange(nVisits),all_happybias[s,:],marker='.', ms=20,color=colors[style],alpha=0.5)
	else:
		style = 1
		plt.plot(np.arange(nVisits),all_happybias[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(all_happybias[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(all_happybias[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(all_happybias[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(all_happybias[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Mid NF', 'Post NF', '1M FU'))
plt.xlabel('Visit')
plt.ylabel('A` neutral - A` happy')
plt.title('Negative = better with happy')
plt.ylim([-.3, 0.3])
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
	if subjects[s] < 100:
		style = 0
		plt.plot(np.arange(nVisits),all_neutralface[s,:],marker='.', ms=20,color=colors[style],alpha=0.5)
	else:
		style = 1
		plt.plot(np.arange(nVisits),all_neutralface[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(all_neutralface[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(all_neutralface[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(all_neutralface[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(all_neutralface[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Mid NF', 'Post NF', '1M FU'))
plt.xlabel('Visit')
plt.ylabel('A` neutral face')
#plt.ylim([-.3, 0.3])
plt.title('Positive = better')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,7))
# plot for each subject
for s in np.arange(nsubs):
	if subjects[s] < 100:
		style = 0
		plt.plot(np.arange(nVisits),all_neutralscene[s,:],marker='.', ms=20,color=colors[style],alpha=0.5)
	else:
		style = 1
		plt.plot(np.arange(nVisits),all_neutralscene[s,:], marker='.',ms=20,color=colors[style],alpha=0.5)
plt.errorbar(np.arange(nVisits),np.nanmean(all_neutralscene[HC_ind,:],axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(all_neutralscene[HC_ind,:],axis=0,nan_policy='omit'), label='HC')
plt.errorbar(np.arange(nVisits),np.nanmean(all_neutralscene[MDD_ind,:],axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(all_neutralscene[MDD_ind,:],axis=0,nan_policy='omit'), label='MDD')
plt.xticks(np.arange(nVisits),('Pre NF', 'Mid NF', 'Post NF', '1M FU'))
plt.xlabel('Visit')
#plt.ylim([-.3, 0.3])
plt.ylabel('A` neutral scene')
plt.title('Positive = better')
plt.legend()
plt.show()

# also do just a prime in general over days with neutral attend faces neutral attend scenes
# see if there are just attentional improvements in general
