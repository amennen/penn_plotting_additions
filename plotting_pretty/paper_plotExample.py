
import os
import glob
import argparse
import numpy as np  # type: ignore
import sys
# Add current working dir so main can be run from the top level rtAttenPenn directory
sys.path.append(os.getcwd())
import rtfMRI.utils as utils
import rtfMRI.ValidationUtils as vutils
from rtfMRI.RtfMRIClient import loadConfigFile
from rtfMRI.Errors import ValidationError
from rtAtten.RtAttenModel import getSubjectDayDir
from sklearn.model_selection import KFold
#rom sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from rtfMRI.StructDict import StructDict, MatlabStructDict
from sklearn.metrics import roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
import scipy
font = {'size'   : 22}
# plt.rc('axes', linewidth=2)
matplotlib.rc('font', **font)

import csv
from anne_additions.plotting_pretty.commonPlotting import *
from anne_additions.aprime_file import aprime,get_blockType ,get_blockData_realtime,getImgProp,get_blockData, get_attCategs
# for each subject, you need to run getcs.py in anne_additions first to get cs evidence for that subject
# have python and matlab versions--let's start with matlab 
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


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def transition_matrix(transitions,nstates):
    #n = 1+ max(transitions) #number of states
    n=nstates
    M = [[0]*n for _ in range(n)]
    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

def transition_matrix_shift(transitions,nstates,nshift):
    n=nstates
    M = [[0]*n for _ in range(n)]
    for (i,j) in zip(transitions,transitions[nshift:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

def transition_matrix_shift_average_previous_states(transitions,nstates,nshift):
    "here average transition in state 0:3 and round"
    n=nstates
    M = [[0]*n for _ in range(n)]
    average_states = [-1]*len(transitions)
    for t in np.arange(len(transitions)):
        if t < len(transitions) - 3:
            average_states[t] = int(round(np.mean(transitions[t:t+4])))
        else:
            average_states[t] = int(round(np.mean(transitions[t:])))
    for (i,j) in zip(average_states,transitions[nshift:]):
        M[i][j] += 1
    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M,average_states


def transition_matrix_shift_average(transitions,averages,nstates):
    n=nstates
    M = [[0]*n for _ in range(n)]
    for (i,j) in zip(averages,transitions[2:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

def calculateTrailingAverage(pos):
    n_samples = len(pos)
    ntrials=25
    pos_by_block = np.reshape(pos,(int(n_samples/ntrials),ntrials)) # this is in blocks x trials

    nBlocks = int(n_samples/ntrials)
    all_smoothed_vals = np.zeros((nBlocks,ntrials))
    for b in np.arange(nBlocks):
        this_block = pos_by_block[b,:]
        all_smoothed_vals[b,0] = this_block[0]
        for i in np.arange(ntrials):
            if i==1:
                all_smoothed_vals[b,i] = np.mean([this_block[0:i+1]])
            elif i > 1:
                all_smoothed_vals[b,i] = np.mean(this_block[i-2:i+1])
    vec = all_smoothed_vals.flatten()
    return vec

def convertOpacityToClassification(opacityVals):
    gain=2.3
    x_shift=.2
    y_shift=.12
    steepness=.9
    classificationVals = (1/gain)*(-1)*np.log((steepness/(opacityVals-y_shift)) - 1) + x_shift
    return classificationVals

def getOpacityFromFile(subjectNum,day,run,block):
  """day, run, block are in MATLAB 1-based indices, but block is the run starting with 1 as first real-time run"""
  rtAttenPath = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'
  subjectDir = rtAttenPath + '/' + 'subject' + str(subjectNum)
  dayDir = subjectDir + '/' + 'day' + str(day)
  runDir = dayDir + '/' + 'run' + str(run)
  file = glob.glob(os.path.join(runDir,'blockdata_*.mat'))[0]
  d = utils.loadMatFile(file)
  all_opacity = d.blockData.smoothAttImgProp
  block_opacity = all_opacity[:,block-1 +4][0][0]
  return block_opacity


def getCSFromFile(subjectNum):
  rtAttenPath = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'
  subjectDir = rtAttenPath + '/' + 'subject' + str(subjectNum)
  outfile = subjectDir + '/' 'offlineAUC_RTCS.npz' 
  z=np.load(outfile)
  CS = z['csOverTime']
  return CS



bins = [-1.   , -0.975, -0.9, -0.8 ,-0.7,-0.55,-0.4,-0.2,0,0.2,0.4,0.55,0.7, 0.8 ,  0.9 , 0.975, 1. ]
nbins=len(bins)
subjectNum=114
CS = getCSFromFile(subjectNum)
  
run = 5
day = 0
blockNum=3 # three out of 4
nTRpBlock=25
rt = CS[run,blockNum*nTRpBlock:(blockNum+1)*nTRpBlock,day]
plt.plot(rt)
#plt.plot(CS[run,:,day])
plt.show()

# 1,1 was initial conditions
#run=1 # in this the 0 = neurofeedback 1 (run 2) #when r=1, that's the second neurofeedback run-->run 3
#day=1

#rt = CS[run,25:50,day]
op = getOpacityFromFile(subjectNum,day+1,run+2,blockNum+1)
nTR = len(rt)

fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
#plt.figure(figsize=(12, 9))
#plt.subplots(2,3,1)
plt.plot(np.arange(nTR)+3,rt,color='k',lw=7)
plt.ylim([-1,1])
plt.xlim([1,nTR])
#plt.xlabel('tr (2.0 s)',fontsize=10)
for b in np.arange(len(bins)):
    # plot dashed for each bin
    plt.plot([0, nTR+1],[bins[b],bins[b]],'--',color='k',alpha=0.4)
# plt.ylabel('scene - face\nrepresentation',fontsize=45)
# plt.xlabel('TR (2 s)',fontsize=45)
plt.ylabel('')
plt.xlabel('')
plt.xticks([])
plt.yticks([])
#labels_pos_v = np.array([-1,0,1])
#labels_pos = labels_pos_v.astype(np.str)
#plt.yticks(labels_pos_v,labels_pos,fontsize=30)
#plt.xticks(fontsize=30)
plt.savefig('thesis_plots_checked/paper_clf.eps')
#plt.show()

# fig,ax = plt.subplots(figsize=(12,9))
# sns.despine()
# plt.plot(np.arange(nTR)+1,op, color='k',lw=7)
# plt.xlim([1,nTR])
# plt.ylim([0,1])
# plt.xlabel('tr (2 s)',fontsize=40)
# plt.ylabel('scene opacity',fontsize=40)
# labels_pos_v = np.array([0,1])
# labels_pos = labels_pos_v.astype(np.str)
# plt.yticks(labels_pos_v,labels_pos,fontsize=30)
# plt.xticks(fontsize=30)
# plt.savefig('poster_plots/opacity.png')
# plt.show()

# fig,ax = plt.subplots(figsize=(12,9))
# sns.despine()
# gain=2.3
# x_shift=.2
# y_shift=.12
# steepness=.9
# plt.ylim([0,1])
# x_vals = np.linspace(-1,1,100)
# y_vals = steepness/(1+np.exp(-gain*(x_vals-x_shift)))+y_shift
# plt.plot(x_vals,y_vals, color='k',lw=7)
# plt.yticks([])
# plt.xlim([-1,1])
# plt.xlabel('scene - face',fontsize=40)
# labels_pos_v = np.array([-1,0,1])
# labels_pos = labels_pos_v.astype(np.str)
# plt.xticks(labels_pos_v,labels_pos,fontsize=30)
# plt.ylabel('scene opacity',fontsize=40)
# labels_pos_v = np.array([0,1])
# labels_pos = labels_pos_v.astype(np.str)
# plt.yticks(labels_pos_v,labels_pos,fontsize=30)
# plt.savefig('poster_plots/transfer.png')
# plt.show()
# # for b in np.arange(len(bins)):
# #     # plot dashed for each bin
# #     plt.plot([1.05, 1.2],[bins[b],bins[b]],'--',color='k',alpha=0.3)
# #plt.xlabel('scene opacity',fontsize=10)
# #plt.xticks(np.array([0,0.5,1]),['0','0.5','1'],fontsize=20)

# plt.subplot(2,3,3)
# sns.despine()
# # now create histogram
# n = plt.hist(rt,bins=bins,orientation='horizontal',align='mid')
# labels_pos = np.array([0,6.25])/nTR
# labels_pos = labels_pos.astype(np.str)
# plt.xlim([0,6.25])
# plt.xticks(np.array([0,6.25]),labels_pos,fontsize=20)
# plt.xlabel('p(scene - face classification)',fontsize=10)
# bin_avg = np.array([(a + b) / 2 for a, b in zip(bins, bins[1:])])
# labels_pos = np.array(bins).astype(np.float)
# labels_pos = np.around(labels_pos,decimals=2).astype(np.str)
# #plt.yticks(bins,labels_pos,fontsize=5)
# plt.yticks()
# plt.ylim([-1,1])
# nbins=len(bins)
# # show matrix of probability
# indices = np.digitize(rt,bins)
# indices[np.argwhere(indices==len(bins))] = len(bins) - 1
# indices_0ind = indices.copy() - 1

# plt.subplots_adjust(wspace=0.3)
# plt.show()

# fig,ax = plt.subplots(figsize=(12,9))
# sns.despine()
# nshift1=2
# nshift2=3
# M1 = np.array(transition_matrix_shift(indices_0ind,nbins-1,nshift1))
# M2 = np.array(transition_matrix_shift(indices_0ind,nbins-1,nshift2))
# M_combined = np.concatenate((M1[:,:,np.newaxis],M2[:,:,np.newaxis]),axis=2)
# M = np.mean(M_combined,axis=2)
# plt.imshow(M,cmap='Reds',vmin=0,vmax=1,origin='lower') # for half, max diff is .2, for all days, max diff is 0.1
# labels_pos = np.array(bins).astype(np.float)
# labels_pos = np.around(labels_pos,decimals=2).astype(np.str)
# plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=10)
# plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=10)
# plt.xlabel('scene - face (t + 5 s)',fontsize=40)
# plt.ylabel('scene - face (t)',fontsize=40)
# #plt.title('p(b|a)')
# #plt.colorbar()
# plt.savefig('poster_plots/ex_matrix_nocolorbar.png')
# plt.show()

### NEW: make square version of indices plotted
# make matrix for imshow!
indices = np.digitize(rt,bins)
indices[np.argwhere(indices==len(bins))] = len(bins) - 1
indices_0ind = indices.copy() - 1
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
all_states = np.zeros((nbins-1,nTR+3))
for t in np.arange(nTR):
  this_state = indices_0ind[t]
  n_this_state = len(np.argwhere(indices_0ind==this_state))
  all_states[this_state,t+3] = 1
  #p_this_state = n_this_state/nTR
  #all_states[this_state,t] = p_this_state
plt.imshow(all_states,cmap='Greys',vmin=0,vmax=1,origin='lower')
plt.ylim([0,nbins-1])
plt.xlim([1,nTR])
# plt.ylabel('scene - face\nrepresentation',fontsize=45)
# plt.xlabel('TR (2 s)',fontsize=45)
plt.ylabel('')
plt.xlabel('')
# labels_pos_v = np.array([-1,1])
# labels_pos = labels_pos_v.astype(np.str)
plt.yticks([])
plt.xticks([])
plt.savefig('thesis_plots_checked/paper_binned_clf.eps')
plt.show()