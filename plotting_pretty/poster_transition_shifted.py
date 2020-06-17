 #plot results/look at group differences

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
font = {'weight' : 'normal',
        'size'   : 22}
import csv
from anne_additions.plotting_pretty.commonPlotting import *
from anne_additions.aprime_file import aprime,get_blockType ,get_blockData_realtime,getImgProp,get_blockData, get_attCategs
matplotlib.rc('font', **font)
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

subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101, 102,103,104,105,106, 107,108,109,110,111,112,113,114,115])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
nsubs = len(subjects)
d1_runs = 6
d2_runs = 8
d3_runs = 7
totalRuns = d1_runs + d2_runs + d3_runs
# TO DO : TAKE INITIAL, MIDDLE, AND END AUC!! use that for controls 
takeHalf = 1
takeAverage = 0
useOpacity = 0
nshift1=2
nshift2=3
comebineShift=1 # then use shift 2 as well, otherwise just do 1

all_x = {}
all_opacity = {}
rtAttenPath = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'
master_x = []
master_opacity=[]
# get HC averages for each RUN OF SCANNER/DAY
nDays = 3

for s in np.arange(nsubs):
    subject_key = 'subject' + str(subjects[s])
    all_x[subject_key] = {}
    all_opacity[subject_key] = {}
    subjectDir = rtAttenPath + '/' + 'subject' + str(subjects[s])
    outfile = subjectDir + '/' 'offlineAUC_RTCS.npz'    
    z=np.load(outfile)
    if subjects[s] == 106:
        d1_runs = 5
    else:
        d1_runs = 6
    CS = z['csOverTime']
    nTR = np.shape(CS)[1]
    for d in np.arange(nDays):
        day_x = []
        day_opacity = []
        if d == 0:
            if takeHalf:
                d1_runs = 3
            categSep = CS[0:d1_runs,:,0]
            nRuns = np.shape(categSep)[0]
            if useOpacity:
                opacity_mat = np.zeros((nRuns,100))
                for r in np.arange(nRuns):
                    opacity_mat[r,:] = getImgProp(subjects[s],d+1,r+2)
        elif d == 1:
            if takeHalf:
                runs_taken = np.array([2,3,4])
            else:
                runs_taken = np.arange(d2_runs)
            categSep = CS[runs_taken,:,1]
            nRuns = np.shape(categSep)[0]
            if useOpacity:
                opacity_mat = np.zeros((nRuns,100))
                for r in np.arange(nRuns):
                    opacity_mat[r,:] = getImgProp(subjects[s],d+1,runs_taken[r]+2)
        elif d == 2:
            if takeHalf:
                runs_taken = np.arange(4,d3_runs)
            else:
                runs_taken = np.arange(d3_runs)
            categSep = CS[runs_taken,:,2]
            nRuns = np.shape(categSep)[0]
            if useOpacity:
                opacity_mat = np.zeros((nRuns,100))
                for r in np.arange(nRuns):
                    opacity_mat[r,:] = getImgProp(subjects[s],d+1,runs_taken[r]+2)
        vec = categSep.flatten()
        day_key = 'day' + str(d)
        if useOpacity:
            vec_opacity = opacity_mat.flatten()
            vec_opacity_classification = convertOpacityToClassification(vec_opacity)
            day_opacity.append(vec_opacity_classification)
            master_opacity.extend(day_opacity[0])
            all_opacity[subject_key][day_key] = day_opacity
        else:
            all_opacity[subject_key][day_key] = [0]
        day_x.append(vec)
        all_x[subject_key][day_key] = day_x
        master_x.extend(day_x[0])
# first make giant histogram


# np.arange(-1,-0.75,0.025)
# np.arange(0.8,1.05, 0.025)
bins = [-1.   , -0.975, -0.9, -0.8 ,-0.7,-0.55,-0.4,-0.2,0,0.2,0.4,0.55,0.7, 0.8 ,  0.9 , 0.975, 1. ]
nbins=len(bins)
# plt.figure()
# n = plt.hist(master_x,bins)
labels_pos = np.array(bins).astype(np.float)
labels_pos = np.around(labels_pos,decimals=2).astype(np.str)
# plt.xticks(bins,labels_pos,fontsize=8)
# plt.ylabel('counts in range')
# plt.show()

# if useOpacity:
#     np.arange(-1,-0.75,0.025)
#     np.arange(0.8,1.05, 0.025)
#     bins = [-1.   , -0.975, -0.9, -0.8 ,-0.7,-0.55,-0.4,-0.2,0,0.2,0.4,0.55,0.7, 0.8 ,  0.9 , 0.975, 1. ]
#     plt.figure()
#     n = plt.hist(master_opacity,bins)
#     labels_pos = np.array(bins).astype(np.float)
#     labels_pos = np.around(labels_pos,decimals=2).astype(np.str)
#     plt.xticks(bins,labels_pos,fontsize=8)
#     plt.ylabel('counts in range')
#     plt.show()


# should probably just remove beginning TRs
# now separate

# after meeting with Ken on 11/7 --> change distribution to be uniform
# first pass: use everyones 
#step_size=0.2
#pos_edges = np.arange(-1,1+step_size,step_size)
#nbins = len(pos_edges)
all_matrices = np.zeros((nbins-1,nbins-1,nsubs,nDays))
p_state = np.zeros((nsubs,nbins-1,nDays))
subject_averages = np.zeros((nsubs,nDays))
for s in np.arange(nsubs):
    subject_key = 'subject' + str(subjects[s])
    for d in np.arange(nDays):
        day_key = 'day' + str(d)
        pos = all_x[subject_key][day_key][0]
        subject_averages[s,d] = np.mean(pos)
        opacity = all_opacity[subject_key][day_key][0]
        #indices = np.digitize(pos,pos_edges)
        indices = np.digitize(pos,bins)
        # if evidence is exactly 1.0 will be outside range, so scale back to 10
        #indices[np.argwhere(indices==len(pos_edges))] = len(pos_edges) - 1
        indices[np.argwhere(indices==len(bins))] = len(bins) - 1
        indices_0ind = indices.copy() - 1
        for st in np.arange(nbins-1):
            n_this_state = len(np.argwhere(indices_0ind==(st)))
            if n_this_state > 0:
                p_state[s,st,d] = n_this_state/len(indices_0ind)

        if takeAverage:
            avgs = calculateTrailingAverage(pos)
            avg_indices = np.digitize(avgs,bins)
            avg_indices[np.argwhere(avg_indices==len(bins))] = len(bins) - 1
            avg_indices_0ind = avg_indices.copy() - 1
            # now make sure from 0 - 9
            # want to sort into bin[x] --> see what 
            #M = np.array(transition_matrix_shift(indices_0ind,nbins-1))
            M = np.array(transition_matrix_shift_average(indices_0ind,avg_indices_0ind,nbins-1))

            all_matrices[:,:,s,d] = M # [1:,1:] - now made index 0 based so don't need to do this
            if len(np.unique(avg_indices_0ind)) != nbins-1:
                print('subject %i, day %i' % (s,d))
                print('len n states = %i' % len(np.unique(avg_indices_0ind)))
                values_taken = np.unique(avg_indices_0ind)
                other = [x for x in np.arange(nbins-1) if x not in values_taken]
                nbad = len(other)
                for ibad in np.arange(nbad):
                    all_matrices[other[ibad],:,s,d] = np.nan
        else:
            if useOpacity:
                op_indices = np.digitize(opacity,bins)
                op_indices[np.argwhere(avg_indices==len(bins))] = len(bins) - 1
                op_indices_0ind = avg_indices.copy() - 1
                M = np.array(transition_matrix_shift_average(indices_0ind,op_indices_0ind,nbins-1))

                all_matrices[:,:,s,d] = M # [1:,1:] - now made index 0 based so don't need to do this
                if len(np.unique(op_indices_0ind)) != nbins-1:
                    print('subject %i, day %i' % (s,d))
                    print('len n states = %i' % len(np.unique(avg_indices_0ind)))
                    values_taken = np.unique(avg_indices_0ind)
                    other = [x for x in np.arange(nbins-1) if x not in values_taken]
                    nbad = len(other)
                    for ibad in np.arange(nbad):
                        all_matrices[other[ibad],:,s,d] = np.nan
            else:    
            
                #M = np.array(transition_matrix_shift(indices_0ind,nbins-1,nshift))
                M1 = np.array(transition_matrix_shift(indices_0ind,nbins-1,nshift1))
                if comebineShift:
                    M2 = np.array(transition_matrix_shift(indices_0ind,nbins-1,nshift2))
                    M_combined = np.concatenate((M1[:,:,np.newaxis],M2[:,:,np.newaxis]),axis=2)
                    M = np.mean(M_combined,axis=2)
                else:
                    M = M1
                #M,average_states = np.array(transition_matrix_shift_average_previous_states(indices_0ind,nbins-1,nshift))
                all_matrices[:,:,s,d] = M # [1:,1:] - now made index 0 based so don't need to do this
                indices_that_matter = indices_0ind.copy()
                #indices_that_matter = average_states.copy()
                if len(np.unique(indices_that_matter)) != nbins-1:
                    print('subject %i, day %i' % (s,d))
                    print('len n states = %i' % len(np.unique(indices_that_matter)))
                    values_taken = np.unique(indices_that_matter)
                    other = [x for x in np.arange(nbins-1) if x not in values_taken]
                    nbad = len(other)
                    for ibad in np.arange(nbad):
                        all_matrices[other[ibad],:,s,d] = np.nan
        # check that the matrix probabilities sum to 1
        this_sum = np.nansum(all_matrices[:,:,s,d],axis=1)
        err=1E-10
        if len(np.argwhere(np.abs(this_sum-1) > err)):
            print('subject %i, day %i' % (s,d))
            print('BAD - NOT SUMMING TO 1')
        # check histogram for each subject
        #plt.figure()
        #plt.hist(pos,bins)
        #plt.show()
        #for row in M: print(' '.join('{0:.2f}'.format(x) for x in row))
        #for row in all_matrices[:,:,s,d]: print(' '.join('{0:.2f}'.format(x) for x in row))




# plot diff of upper square both groups over 1 --> 3
i=0
j=0
# i,j 0.2 is for -1-->-0.9
# try to stay positive = 15 15
data_mat = all_matrices[i,j,:,:]
fig = plotPosterStyle(data_mat,subjects)
plt.ylim([0,1])
x,y = nonNan(data_mat[MDD_ind,0],data_mat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat(p/2,0,2,np.nanmax(data_mat),0.05)
x,y = nonNan(data_mat[HC_ind,0],data_mat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(data_mat),0.01)
x,y = nonNan(data_mat[HC_ind,2],data_mat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,2,np.nanmax(data_mat),0.01)
plt.show()


i=0
j=0
stat = all_matrices[i,j,:,:]
fig,ax = plotPosterStyle_DF(stat,subjects)
plt.ylim([0,1])
plt.xticks(np.arange(3),('NF 1', 'NF 2', 'NF 3'))
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.01)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,2,np.nanmax(stat),0.01)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat(p/2,0,2,np.nanmax(stat),0.05)
plt.show()

stat = subject_averages
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,2])],subjects)
x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD_1 > HC_1$')
x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,1.2,np.nanmax(stat)+.2,0.05,0,'$MDD_1 > MDD_3$')
plt.ylabel('average classification',fontsize=40)
plt.ylim([-.2,.7])
plt.xticks(np.arange(2),('NF 1','NF 3'),fontsize=30)
#labels_pos_v = np.array([0,0.5,1])
#labels_pos = labels_pos_v.astype(np.str)
plt.yticks(fontsize=30)
plt.xlabel('day',fontsize=35)
plt.show()

################### DIAGONAL ONLY NOW ##############
diagonal_data = np.zeros((nsubs,nbins-1,nDays))
labels_pos = np.array(bins).astype(np.float)
labels_pos = np.around(labels_pos,decimals=2).astype(np.str)
for s in np.arange(nsubs):
    for d in np.arange(nDays):
        ex = all_matrices[:,:,s,d]
        diagonal = np.diagonal(ex)
        diagonal_data[s,:,d] = diagonal
# now plot for day 1 
fig = plotPosterStyle_multiplePTS(diagonal_data[:,:,np.array([0,2])],subjects)
plt.subplot(1,2,1)
plt.ylim([0,.8])
plt.yticks(np.linspace(0,.75,4),fontsize=25)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xlabel('bin at t')
plt.ylabel('p(stay in bin) at t+5 s')
plt.title('Early NF')
x,y = nonNan(diagonal_data[MDD_ind,0,0],diagonal_data[HC_ind,0,0])
t,p = scipy.stats.ttest_ind(x,y)
#addSingleStat(p/2,0,np.nanmax(triangle_data),0.01)
plt.subplot(1,2,2)
#plt.ylim([0,.4])
plt.ylim([0,.8])
plt.yticks(np.linspace(0,.75,4),fontsize=25)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xlabel('bin at t')
plt.title('Late NF')
# plt.subplot(1,3,3)
# #plt.ylim([0,.4])
# plt.yticks([])
# plt.xlabel('bin group')
plt.show()

stat = p_state[:,0,:]
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,2])],subjects)
x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD_1 > HC_1$')
x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,1.2,np.nanmax(stat)+.2,0.05,0,'$MDD_1 > MDD_3$')
plt.ylabel('p(saddest state)',fontsize=40)
plt.ylim([0,.6])
plt.xticks(np.arange(2),('NF 1','NF 3'),fontsize=30)
#labels_pos_v = np.array([0,0.5,1])
#labels_pos = labels_pos_v.astype(np.str)
plt.yticks(fontsize=30)
plt.xlabel('day',fontsize=35)
plt.show()

stat = p_state[:,15,:]
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,2])],subjects)
x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD_1 < HC_1$' )
x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0,'$MDD_1 < HC_1$' )
x,y = nonNan(stat[HC_ind,0],stat[HC_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,-0.2,0.8,np.nanmax(stat)+0.2,0.05,0,'$HC_1 < HC_3$')
plt.ylabel('p(correct state)',fontsize=40)
plt.ylim([0,.6])
plt.xticks(np.arange(2),('NF 1','NF 3'),fontsize=30)
#labels_pos_v = np.array([0,0.5,1])
#labels_pos = labels_pos_v.astype(np.str)
plt.yticks(fontsize=30)
plt.xlabel('day',fontsize=35)
plt.show()



# make same diagonal plot but with p_state
fig = plotPosterStyle_multiplePTS(p_state[:,:,np.array([0,2])],subjects)
plt.subplot(1,2,1)
plt.ylim([0,.25])
#plt.yticks(np.linspace(0,.75,4),fontsize=25)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xlabel('bin at t')
plt.ylabel('p(bin)')
plt.title('Early NF')
x,y = nonNan(p_state[MDD_ind,0,0],p_state[HC_ind,0,0])
t,p = scipy.stats.ttest_ind(x,y)
#addSingleStat(p/2,0,0.15,0.01)
plt.subplot(1,2,2)
#plt.ylim([0,.4])
plt.ylim([0,.25])
#plt.yticks(np.linspace(0,.75,4),fontsize=25)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xlabel('bin at t')
plt.title('Late NF')
# plt.subplot(1,3,3)
# #plt.ylim([0,.4])
# plt.yticks([])
# plt.xlabel('bin group')
plt.show()
# plot difference of MADRS vs. others
M = getMADRSscoresALL()
d1,d2,d3 = getMADRSdiff(M,subjects)
all_neg_change = p_state[:,0,2] - p_state[:,0,0]
all_neg_change = -1*(subject_averages[:,2] - subject_averages[:,0]) # make this consistent with others so multiply 
# by -1 here so we don't have to changen everything afterwards

colors = ['k', 'r'] # HC, MDD
colors = ['#636363','#de2d26']
#fig = plt.figure(figsize=(10,7))
fig,ax = plt.subplots(figsize=(12,10))
sns.despine()
for s in np.arange(nsubs):
  subjectNum  = subjects[s]
  madrs_change = d2[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(-1*all_neg_change[s],-1*d1[s],marker='.',ms=30,color=colors[style])
#plt.xlabel('improvement in negative stickiness',fontsize=40)
plt.xlabel('improvement in scene classification',fontsize=40)
plt.ylabel('improvement in MADRS',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim([-.5,.8])
plt.title('')
x,y = nonNan(-1*all_neg_change,-1*d1)
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.12,20,text, ha='left',va='top',color='k',fontsize=25)
x,y = nonNan(-1*all_neg_change[MDD_ind],-1*d1[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.12,15,text, ha='left',va='top',color='k',fontsize=25)
# labels_pos_v = np.array([-0.4,0,0.4,0.8])
# labels_pos = labels_pos_v.astype(np.str)
# plt.xticks(labels_pos_v,labels_pos,fontsize=30)
#plt.savefig('poster_plots/MADRS_v_NF.png')
plt.show()






# plot difference of MADRS vs. others
M = getMADRSscoresALL()
d1,d2,d3 = getMADRSdiff(M,subjects)
data_mat = all_matrices[0,0,:,:]
all_neg_change = data_mat[:,2] - data_mat[:,0]
#all_neg_change = specifically_neg
colors = ['k', 'r'] # HC, MDD
colors = ['#636363','#de2d26']
#fig = plt.figure(figsize=(10,7))
fig,ax = plt.subplots(figsize=(12,10))
sns.despine()
for s in np.arange(nsubs):
#for s in keep:
  subjectNum  = subjects[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(-1*all_neg_change[s],-1*d1[s],marker='.',ms=30,color=colors[style])
plt.xlabel('improvement in negative stickiness',fontsize=40)
plt.ylabel('improvement in depression severity',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim([-.5,.8])
plt.title('Transitions %i shifted ahead' % nshift1)
x,y = nonNan(-1*all_neg_change,-1*d1)
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.47,20,text, ha='left',va='top',color='k',fontsize=25)
x,y = nonNan(-1*all_neg_change[MDD_ind],-1*d1[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.47,15,text, ha='left',va='top',color='k',fontsize=25)
labels_pos_v = np.array([-0.4,0,0.4,0.8])
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(labels_pos_v,labels_pos,fontsize=30)
#plt.savefig('poster_plots/MADRS_v_NF.png')
plt.show()

# can also look at - does is change predicted by how much you start with?
pos_improvement = all_matrices[15,15,:,2] - all_matrices[15,15,:,0]
all_neg_change = all_matrices[0,0,:,2] - all_matrices[0,0,:,0]
specifically_neg = -1*all_neg_change/( -1*all_neg_change+ pos_improvement)
p_diff = p_state[:,0,2] - p_state[:,0,0]
fig,ax = plt.subplots(figsize=(12,10))
sns.despine()
for s in np.arange(nsubs):
  subjectNum  = subjects[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(pos_improvement[s],-1*all_neg_change[s],marker='.',ms=30,color=colors[style])
plt.xlabel('improvement in general prob',fontsize=40)
plt.ylabel('improvement in stickiness',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.plot([.8,-.8],[.8,-.8],linewidth=2,color='k')
# plt.ylim([-.6,.6])
# plt.xlim([-.6,.6])
#plt.xlim([-.5,.8])
#plt.title('Transitions %i shifted ahead' % nshift1)
x,y = nonNan(pos_improvement,-1*all_neg_change)
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(.47,.5,text, ha='left',va='top',color='k',fontsize=25)
x,y = nonNan(-1*all_neg_change[MDD_ind],pos_improvement[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(.47,.35,text, ha='left',va='top',color='k',fontsize=25)
# labels_pos_v = np.array([-0.4,0,0.4,0.8])
# labels_pos = labels_pos_v.astype(np.str)
# plt.xticks(labels_pos_v,labels_pos,fontsize=30)
#plt.savefig('poster_plots/MADRS_v_NF.png')
plt.show()

keep = np.argwhere(-1*all_neg_change > pos_improvement)

### for poster -- only show day 1 and 3
i=0
j=0
stat = all_matrices[i,j,:,:]
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,2])],subjects)

x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD_1 > HC_1$')
x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,1.2,np.nanmax(stat)+.2,0.05,0,'$MDD_1 > MDD_3$')
plt.ylabel('p(stay in saddest state)',fontsize=40)
plt.ylim([0,1.05])
plt.xticks(np.arange(2),('NF 1','NF 3'),fontsize=30)
labels_pos_v = np.array([0,0.5,1])
labels_pos = labels_pos_v.astype(np.str)
plt.yticks(labels_pos_v,labels_pos,fontsize=30)
plt.xlabel('day',fontsize=35)
plt.savefig('poster_plots/NF_saddest_bargraph.png')
plt.show()

i=15
j=15
stat = all_matrices[i,j,:,:]
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,2])],subjects)
x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD_1 < HC_1$' )
x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0,'$MDD_1 < HC_1$' )
x,y = nonNan(stat[HC_ind,0],stat[HC_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,-0.2,0.8,np.nanmax(stat)+0.2,0.05,0,'$HC_1 < HC_3$')
plt.ylabel('p(stay in correct state)',fontsize=40)
plt.ylim([0,1.05])
plt.xticks(np.arange(2),('NF 1','NF 3'),fontsize=30)
labels_pos_v = np.array([0,0.5,1])
labels_pos = labels_pos_v.astype(np.str)
plt.yticks(labels_pos_v,labels_pos,fontsize=30)
plt.xlabel('day',fontsize=35)
plt.savefig('poster_plots/NF_correct_bargraph.png')
plt.show()




# GET GENERAL PROB IN LOWEST STATE
i=15
stat = p_state[i,:,:]
fig,ax = plotPosterStyle_DF(stat,subjects)
#plt.ylim([0,1])
plt.ylabel('p(+1)')
plt.xticks(np.arange(3),('NF 1', 'NF 2', 'NF 3'))
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.01)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,2,np.nanmax(stat),0.01)
x,y = nonNan(stat[HC_ind,0],stat[HC_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat(p/2,0,2,np.nanmax(stat),0.05)
plt.show()


# plot diff of upper square both groups over 1 --> 3
i=15
j=15
stat = all_matrices[i,j,:,:]
fig,ax = plotPosterStyle_DF(stat,subjects)
plt.ylim([0,1])
plt.xticks(np.arange(3),('NF 1', 'NF 2', 'NF 3'))
x,y = nonNan(stat[HC_ind,0],stat[HC_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat(p/2,0,2,np.nanmax(stat),0.05)
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.01)
x,y = nonNan(stat[HC_ind,2],stat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,2,np.nanmax(stat),0.01)
plt.show()

########## CONTROL IMPROVEMENT #################
improvement = all_matrices[15,15,:,2] - all_matrices[15,15,:,0]
AUC_change = day_average[:,2] - day_average[:,0]
colors = ['#636363','#de2d26']
plt.figure()
for s in np.arange(nsubs):
  subjectNum  = subjects[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(AUC_change[s],improvement[s],marker='.',ms=20,color=colors[style])
plt.show()
x,y = nonNan(AUC_change[HC_ind],improvement[HC_ind])
scipy.stats.pearsonr(x,y)
########## CONTROL IMPROVEMENT #################




x,y = nonNan(-1*d1[MDD_ind],all_neg_change[MDD_ind])
scipy.stats.pearsonr(x,y)
x,y = nonNan(-1*d2[MDD_ind],all_neg_change[MDD_ind])
scipy.stats.pearsonr(x,y)
x,y = nonNan(-1*d1,all_neg_change)
scipy.stats.pearsonr(x,y)

##### WHEN YOU RUN THE AMYGALA SCRIPT ########### WHEN YOU RUN THE AMYGALA SCRIPT ######
amyg= all_subject_averages_fearful[:,1,:] - all_subject_averages_fearful[:,0,:]
amyg_change = amyg[:,1] - amyg[:,0]
#amyg_change = all_subject_averages_fearful[:,0,1] - all_subject_averages_fearful[:,0,0]
data_mat = all_matrices[0,0,:,:]
all_neg_change = data_mat[:,2] - data_mat[:,0]
colors = ['#636363','#de2d26']
fig,ax = plt.subplots(figsize=(13,10))
sns.despine()
for s in np.arange(nsubs):
  subjectNum  = subjects[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(-1*all_neg_change[s],-1*amyg_change[s],marker='.',ms=30,color=colors[style])
plt.xlabel('improvement in negative stickiness',fontsize=40)
plt.ylabel('reduction in LA reactivity',fontsize=40)
x,y = nonNan(-1*all_neg_change,-1*amyg_change)
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.47,.8,text, ha='left',va='top',color='k',fontsize=22)
x,y = nonNan(-1*all_neg_change[MDD_ind],-1*amyg_change[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.47,.6,text, ha='left',va='top',color='k',fontsize=22)
plt.xlim([-.5,.8])
plt.ylim([-.4,.8])
labels_pos_v = np.array([-0.4,0,0.4,0.8])
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(labels_pos_v,labels_pos,fontsize=30)
plt.yticks(labels_pos_v,labels_pos,fontsize=30)
plt.savefig('poster_plots/LA_v_NF.png')
plt.show()


##### WHEN YOU RUN THE AMYGALA SCRIPT ########### WHEN YOU RUN THE AMYGALA SCRIPT ######
data_mat = all_matrices[15,15,:,:]
all_pos_change = data_mat[:,2] - data_mat[:,0]
#all_pos_change = p_state[15,:,2] - p_state[15,:,0]
all_attention_change = combined_avg[:,2] - combined_avg[:,0]
colors = ['#636363','#de2d26']
fig = plt.figure(figsize=(10,7))
for s in np.arange(nsubs):
  subjectNum  = subjects[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(all_pos_change[s],all_attention_change[s],marker='.',ms=20,color=colors[style])
plt.xlabel('Improvement in sustained attention 3 - 1')
plt.ylabel('Improvement in A'' behavior 3 - 1')
#plt.xlim([-0.4,0.4])
plt.show()
x,y = nonNan(all_pos_change,all_attention_change)
scipy.stats.pearsonr(x,y)


# start with a matrix -- TRIANGLE ANALYSIS
nTriangles = 5
triangle_data = np.zeros((nsubs,nTriangles,nDays))
for s in np.arange(nsubs):
    for d in np.arange(nDays):
        ex = all_matrices[:,:,s,d]
        diagonal = np.diagonal(ex)
        diagonal_m1 = np.diagonal(ex,offset=-1)
        diagonal_m2 = np.diagonal(ex,offset=-2)
        diagonal_m3 = np.diagonal(ex,offset=-3)
        triangle_data[s,0,d] = np.mean(np.concatenate([diagonal[0:3].flatten(),diagonal_m1[0:2].flatten(),diagonal_m2[0].flatten()]))
        triangle_data[s,1,d] = np.mean(np.concatenate([diagonal[3:6].flatten(),diagonal_m1[3:5].flatten(),diagonal_m2[3].flatten()]))
        triangle_data[s,2,d] = np.mean(np.concatenate([diagonal[6:10].flatten(),diagonal_m1[6:9].flatten(),diagonal_m2[6:8].flatten(),diagonal_m3[6].flatten()]))
        triangle_data[s,3,d] = np.mean(np.concatenate([diagonal[10:13].flatten(),diagonal_m1[10:12].flatten(),diagonal_m2[10].flatten()]))
        triangle_data[s,4,d] = np.mean(np.concatenate([diagonal[13:16].flatten(),diagonal_m1[13:15].flatten(),diagonal_m2[13].flatten()]))
# now do multiple point plot
fig = plotPosterStyle_multiplePTS(triangle_data,subjects)
plt.subplot(1,3,1)
plt.ylim([0,.4])
plt.yticks(np.linspace(0,.4,5),fontsize=15)
plt.xlabel('triangle group')
x,y = nonNan(triangle_data[MDD_ind,0,0],triangle_data[HC_ind,0,0])
t,p = scipy.stats.ttest_ind(x,y)
#addSingleStat(p/2,0,np.nanmax(triangle_data),0.01)
plt.subplot(1,3,2)
plt.ylim([0,.4])
plt.yticks([])
plt.xlabel('triangle group')
plt.subplot(1,3,3)
plt.ylim([0,.4])
plt.yticks([])
plt.xlabel('triangle group')
plt.show()


nTriangles = 5
triangle_data = np.zeros((nsubs,nTriangles,nDays))
for s in np.arange(nsubs):
    for d in np.arange(nDays):
        ex = all_matrices[:,:,s,d]
        diagonal = np.diagonal(ex)
        diagonal_m1 = np.diagonal(ex,offset=-1)
        diagonal_m2 = np.diagonal(ex,offset=-2)
        diagonal_m3 = np.diagonal(ex,offset=-3)
        triangle_data[s,0,d] = np.mean(diagonal[0:3].flatten())
        triangle_data[s,1,d] = np.mean(diagonal[3:6].flatten())
        triangle_data[s,2,d] = np.mean(diagonal[6:10].flatten())
        triangle_data[s,3,d] = np.mean(diagonal[10:13].flatten())
        triangle_data[s,4,d] = np.mean(diagonal[13:16].flatten())
# now do multiple point plot
fig = plotPosterStyle_multiplePTS(triangle_data,subjects)
plt.subplot(1,3,1)
plt.ylim([0,.4])
plt.yticks(np.linspace(0,.4,5),fontsize=15)
plt.xlabel('triangle group')
x,y = nonNan(triangle_data[MDD_ind,0,0],triangle_data[HC_ind,0,0])
t,p = scipy.stats.ttest_ind(x,y)
#addSingleStat(p/2,0,np.nanmax(triangle_data),0.01)
plt.subplot(1,3,2)
plt.ylim([0,.4])
plt.yticks([])
plt.xlabel('triangle group')
plt.subplot(1,3,3)
plt.ylim([0,.4])
plt.yticks([])
plt.xlabel('triangle group')
plt.show()


# NOW DON'T INCLUDE DIAGONAL
nTriangles = 5
triangle_data = np.zeros((nsubs,nTriangles,nDays))
for s in np.arange(nsubs):
    for d in np.arange(nDays):
        ex = all_matrices[:,:,s,d]
        diagonal = np.diagonal(ex)
        diagonal_m1 = np.diagonal(ex,offset=-1)
        diagonal_m2 = np.diagonal(ex,offset=-2)
        diagonal_m3 = np.diagonal(ex,offset=-3)
        triangle_data[s,0,d] = np.mean(np.concatenate([diagonal_m1[0:2].flatten(),diagonal_m2[0].flatten()]))
        triangle_data[s,1,d] = np.mean(np.concatenate([diagonal_m1[3:5].flatten(),diagonal_m2[3].flatten()]))
        triangle_data[s,2,d] = np.mean(np.concatenate([diagonal_m1[6:9].flatten(),diagonal_m2[6:8].flatten(),diagonal_m3[6].flatten()]))
        triangle_data[s,3,d] = np.mean(np.concatenate([diagonal_m1[10:12].flatten(),diagonal_m2[10].flatten()]))
        triangle_data[s,4,d] = np.mean(np.concatenate([diagonal_m1[13:15].flatten(),diagonal_m2[13].flatten()]))
# now do multiple point plot
fig = plotPosterStyle_multiplePTS(triangle_data,subjects)
plt.subplot(1,3,1)
plt.ylim([0,.4])
plt.yticks(np.linspace(0,.4,5),fontsize=15)
plt.xlabel('triangle group')
x,y = nonNan(triangle_data[MDD_ind,0,0],triangle_data[HC_ind,0,0])
t,p = scipy.stats.ttest_ind(x,y)
#addSingleStat(p/2,0,np.nanmax(triangle_data),0.01)
plt.subplot(1,3,2)
plt.ylim([0,.4])
plt.yticks([])
plt.xlabel('triangle group')
plt.subplot(1,3,3)
plt.ylim([0,.4])
plt.yticks([])
plt.xlabel('triangle group')
plt.show()



x,y = nonNan(triangle_data[MDD_ind,0,0],triangle_data[MDD_ind,0,2])
t,p = scipy.stats.ttest_rel(x,y)
x,y = nonNan(triangle_data[HC_ind,0,0],triangle_data[HC_ind,0,2])
t,p = scipy.stats.ttest_rel(x,y)

for row in ex: print(' '.join('{0:.2f}'.format(x) for x in row))
np.diagonal(ex)
np.diagonal(ex,offset=-1)
triangle_data[0,:,:] = np.mean(all_matrices[0:3,0:3,:,:])



# if we cared about conditional probabilties here they are
ind_bin=0
this_mat = all_matrices[ind_bin,:,:,:]
new_mat = np.zeros((nsubs,nbins-1,nDays))
for s in np.arange(nsubs):
    new_mat[s,:,:] = this_mat[:,s,:] 
fig = plotPosterStyle_multiplePTS(new_mat[:,:,np.array([0,2])],subjects)

plt.show()


ind_bin=15
data_mat1 = all_matrices[ind_bin,:,:,0]
data_mat2 = all_matrices[ind_bin,:,:,1]
data_mat3 = all_matrices[ind_bin,:,:,2]
bin_avg = np.array([(a + b) / 2 for a, b in zip(bins, bins[1:])])
expectations_mat = np.zeros((nsubs,3))
expectations_mat[:,0] = np.dot(data_mat1.T,bin_avg)
expectations_mat[:,1] = np.dot(data_mat2.T,bin_avg)
expectations_mat[:,2] = np.dot(data_mat3.T,bin_avg)
fig = plotPosterStyle_multiplePTS(expectations_mat[:,np.array([0,2])],subjects)
plt.show()
# plot average transition matrix by day
#labels_pos = pos_edges.astype(np.float)
#labels_pos = np.around(labels_pos,decimals=3).astype(np.str)
labels_pos = np.array(bins).astype(np.float)
labels_pos = np.around(labels_pos,decimals=2).astype(np.str)
vmin=0
vmax=1
#day 1 first plt.figure(figsize=(20,20))
d=0
# make sequential colormap
fig,ax = plt.subplots(figsize=(20,20))
plt.subplot(1,3,1)
this_plot_hc = np.nanmean(all_matrices[:,:,HC_ind,d],axis=2)
plt.imshow(this_plot_hc,cmap='Reds',vmin=vmin,vmax=vmax,origin='lower')
#plt.colorbar()
plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('HC',fontsize=20)
plt.subplot(1,3,2)
this_plot_mdd = np.nanmean(all_matrices[:,:,MDD_ind,d],axis=2)
plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.imshow(this_plot_mdd,cmap='Reds',vmin=vmin,vmax=vmax,origin='lower')
plt.xlabel('value B')
plt.title('MDD',fontsize=20)
#plt.colorbar()
plt.show()
# to do: understand output of transition matrix -- which is A/which is B, plot for each day
#### THIS IS WHAT IS ON THE POSTER ###
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
d=0
this_plot_hc = np.nanmean(all_matrices[:,:,HC_ind,d],axis=2)
this_plot_mdd = np.nanmean(all_matrices[:,:,MDD_ind,d],axis=2)
z=this_plot_mdd-this_plot_hc
np.max(np.abs(z))
plt.imshow(this_plot_mdd-this_plot_hc,cmap='RdGy_r',vmin=-.25,vmax=.25,origin='lower') # for half, max diff is .2, for all days, max diff is 0.1
#plt.title('NF 1')
plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=10)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=10)
plt.xlabel('scene - face (t + 5 s)',fontsize=40)
plt.ylabel('scene - face (t)',fontsize=40)
#plt.title('p(b|a)')
#plt.colorbar()
plt.savefig('poster_plots/matrix_day1.png')
plt.show()
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
d=2
this_plot_hc = np.nanmean(all_matrices[:,:,HC_ind,d],axis=2)
this_plot_mdd = np.nanmean(all_matrices[:,:,MDD_ind,d],axis=2)
z=this_plot_mdd-this_plot_hc
np.max(np.abs(z))
sns.despine()
plt.imshow(this_plot_mdd-this_plot_hc,cmap='RdGy_r',vmin=-.25,vmax=.25,origin='lower') # for half, max diff is .2, for all days, max diff is 0.1
plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=10)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=10)
plt.xlabel('scene - face (t + 5 s)',fontsize=40)
plt.ylabel('scene - face (t)',fontsize=40)
#plt.title('p(b|a)')
#plt.colorbar()
plt.colorbar()
plt.savefig('poster_plots/matrix_day3_colorbar.png')
#plt.ylabel('t')
#plt.title('NF 3')
#plt.title('MDD - HC')
plt.show()

# now do the same by day by group
this_plot_hc1 = np.nanmean(all_matrices[:,:,HC_ind,0],axis=2)
this_plot_hc3 = np.nanmean(all_matrices[:,:,HC_ind,2],axis=2)
this_plot_mdd1 = np.nanmean(all_matrices[:,:,MDD_ind,0],axis=2)
this_plot_mdd3 = np.nanmean(all_matrices[:,:,MDD_ind,2],axis=2)
plt.figure(figsize=(10,10))
plt.imshow(this_plot_hc3-this_plot_hc1,cmap='bwr',vmin=-.15,vmax=.15,origin='lower') # for half, max diff is .16, for all days, max diff is 0.1
plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('HC2 - HC0')
plt.colorbar()
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(this_plot_mdd3-this_plot_mdd1,cmap='bwr',vmin=-.15,vmax=.15,origin='lower') # for half, max diff is .11, for all days, max diff is 0.06
plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('MDD2 - MDD0')
plt.colorbar()
plt.show()



# try to understand how distirbutions are changing--> maybe average shift by group?
# first plot one person's
color_d3_pts = np.array([252 ,174 ,145])/255
color_d3_avg = np.array([203 ,24 ,29])/255
color_d1_pts = np.array([204,204,204])/255
color_d1_avg = np.array([82,82,82])/255
ind_bin = 15
#plt.figure()
fig,ax = plt.subplots()
for s in np.arange(nsubs):
    if s in HC_ind:
        plt.subplot(1,2,1)
        matrix1 = all_matrices[ind_bin,:,s,0]
        matrix3 = all_matrices[ind_bin,:,s,2]
        plt.plot(np.arange(nbins-1),matrix1,color=color_d1_pts,alpha=0.7)
        plt.plot(np.arange(nbins-1),matrix3,color=color_d3_pts,alpha=0.7)
        plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
        plt.xlabel('value B')
        plt.ylabel('p(B|-1)')
        #plt.legend()
    elif s in MDD_ind:
        plt.subplot(1,2,2)
        matrix1 = all_matrices[ind_bin,:,s,0]
        matrix3 = all_matrices[ind_bin,:,s,2]
        plt.plot(np.arange(nbins-1),matrix1,color=color_d1_pts,alpha=0.7)
        plt.plot(np.arange(nbins-1),matrix3,color=color_d3_pts,alpha=0.7)
        plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
        plt.xlabel('value B')
        plt.ylabel('p(B|-1)')
plt.subplot(1,2,1)
hc_m1_avg = np.nanmean(all_matrices[ind_bin,:,HC_ind,0],axis=0)
hc_m3_avg = np.nanmean(all_matrices[ind_bin,:,HC_ind,2],axis=0)
plt.plot(np.arange(nbins-1),hc_m1_avg,color=color_d1_avg,alpha=1,lw=5,label='day 1')
plt.plot(np.arange(nbins-1),hc_m3_avg,color=color_d3_avg,alpha=1, lw=5,label='day 3')
plt.title('HC Group')
plt.ylim([0,1])
plt.legend()
plt.subplot(1,2,2)
mdd_m1_avg = np.nanmean(all_matrices[ind_bin,:,MDD_ind,0],axis=0)
mdd_m3_avg = np.nanmean(all_matrices[ind_bin,:,MDD_ind,2],axis=0)
plt.plot(np.arange(nbins-1),mdd_m1_avg,color=color_d1_avg,alpha=1,lw=5,label='day 1')
plt.plot(np.arange(nbins-1),mdd_m3_avg,color=color_d3_avg,alpha=1, lw=5,label='day 3')
plt.title('MDD Group')
plt.ylim([0,1])
plt.legend()
plt.show()

# NOW: for each subject/day, get expectation of probability distribution
ind_bin=15
data_mat1 = all_matrices[ind_bin,:,:,0]
data_mat2 = all_matrices[ind_bin,:,:,1]
data_mat3 = all_matrices[ind_bin,:,:,2]
bin_avg = np.array([(a + b) / 2 for a, b in zip(bins, bins[1:])])
expectations_mat = np.zeros((nsubs,3))
expectations_mat[:,0] = np.dot(data_mat1.T,bin_avg)
expectations_mat[:,1] = np.dot(data_mat2.T,bin_avg)
expectations_mat[:,2] = np.dot(data_mat3.T,bin_avg)
# relationship in general becomes stronger with looking at first 4 runs, last 4 runs
# then positive relationship is stronger with increase in positive scene activation instead of looking at negative decrease
data_mat = expectations_mat
ndays=3
fig = plotPosterStyle(data_mat,subjects)
#plt.ylim([0,1])
x,y = nonNan(data_mat[MDD_ind,0],data_mat[MDD_ind,ndays-1])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat(p/2,0,ndays-1,np.nanmax(data_mat),0.05)
x,y = nonNan(data_mat[HC_ind,0],data_mat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(data_mat),0.01)
x,y = nonNan(data_mat[HC_ind,ndays-1],data_mat[MDD_ind,ndays-1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,ndays-1,np.nanmax(data_mat),0.01)
plt.show()



# NOW PLOT MDD 3 - MDD 1
mdd_1 = np.nanmean(all_matrices[:,:,MDD_ind,0],axis=2)
mdd_3 = np.nanmean(all_matrices[:,:,MDD_ind,2],axis=2)
plt.figure(figsize=(10,10))
plt.imshow(mdd_3-mdd_1,cmap='bwr',vmin=-0.08,vmax=0.08,origin='lower')
plt.yticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.xticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('MDD_3 - MDD_1')
plt.colorbar()
plt.show()

hc_1 = np.nanmean(all_matrices[:,:,HC_ind,0],axis=2)
hc_3 = np.nanmean(all_matrices[:,:,HC_ind,2],axis=2)
plt.figure(figsize=(10,10))
plt.imshow(hc_3-hc_1,cmap='bwr',vmin=-0.08,vmax=0.08,origin='lower')
plt.yticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.xticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('HC_3 - HC_1')
plt.colorbar()
plt.show()
# newest - get most likely denisty for each subject
# position will span -1 to 1 and velocity can span -2 to 2
most_likely_counts = {}
step_size=0.2
pos_edges = np.arange(-1,1+step_size,step_size)
vel_edges = np.arange(-2,2+step_size*2,step_size*2)
all_hist = np.zeros((len(pos_edges)-1,len(vel_edges)-1,nsubs,nDays))
for s in np.arange(nsubs):
    subject_key = 'subject' + str(subjects[s])
    most_likely_counts[subject_key] = {}
    for d in np.arange(nDays):
        day_key = 'day' + str(d)
        pos = all_x[subject_key][day_key][0]
        vel = all_dx[subject_key][day_key][0]
        H, xedges, yedges = np.histogram2d(pos, vel, bins=(pos_edges, vel_edges),density=False)
        H_norm = H/len(pos)
        all_hist[:,:,s,d] = H_norm
        #plt.imshow(H)
        #plt.show()
        most_counts = np.unravel_index(np.argmax(H),np.shape(H))
        most_likely_counts[subject_key][day_key] = most_counts

# now plot each
# day 1
fig,ax = plt.subplots()
plt.subplot(3,2,1)
this_plot = np.nanmean(all_hist[:,:,HC_ind,0],axis=2)
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.imshow(this_plot,cmap='Blues',origin='lower')
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.subplot(3,2,2)
this_plot = np.nanmean(all_hist[:,:,MDD_ind,0],axis=2)
plt.imshow(this_plot,cmap='Blues',origin='lower')
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.subplot(3,2,3)
this_plot = np.nanmean(all_hist[:,:,HC_ind,1],axis=2)
plt.imshow(this_plot,cmap='Blues',origin='lower')
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.subplot(3,2,4)
this_plot = np.nanmean(all_hist[:,:,MDD_ind,1],axis=2)
plt.imshow(this_plot,cmap='Blues',origin='lower')
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.subplot(3,2,5)
this_plot = np.nanmean(all_hist[:,:,HC_ind,2],axis=2)
plt.imshow(this_plot,cmap='Blues',origin='lower')
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.subplot(3,2,6)
this_plot = np.nanmean(all_hist[:,:,MDD_ind,2],axis=2)
plt.imshow(this_plot,cmap='Blues',origin='lower')
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.colorbar()

plt.show()
labels_vel = vel_edges.astype(np.float)
labels_vel= np.around(labels_vel,decimals=3).astype(np.str)
labels_pos = pos_edges.astype(np.float)
labels_pos = np.around(labels_pos,decimals=3).astype(np.str)
vmin=-0.015
vmax=0.015
#fig,ax = plt.subplots()
#plt.subplot(1,3,1)
plt.figure(figsize=(20,20))
d=0
this_plot_hc = np.nanmean(all_hist[:,:,HC_ind,d],axis=2).T
this_plot_mdd = np.nanmean(all_hist[:,:,MDD_ind,d],axis=2).T
plt.imshow(this_plot_mdd-this_plot_hc,cmap='bwr',vmin=vmin,vmax=vmax,origin='lower')
plt.yticks(np.arange(len(vel_edges))-0.5,labels_vel,fontsize=10)
plt.xticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.ylabel('d(CS)/dt')
plt.xlabel('CS: scene - face evidence',fontsize=15)
plt.title('Day %i' % d)
#plt.axis('equal')
plt.colorbar()

plt.show()
#plt.subplot(1,3,2)
plt.figure(figsize=(20,10))
d=1
this_plot_hc = np.nanmean(all_hist[:,:,HC_ind,d],axis=2).T
this_plot_mdd = np.nanmean(all_hist[:,:,MDD_ind,d],axis=2).T
plt.imshow(this_plot_mdd - this_plot_hc,cmap='bwr',vmin=vmin,vmax=vmax,origin='lower')
plt.yticks(np.arange(len(vel_edges))-0.5,labels_vel,fontsize=10)
plt.xticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.ylabel('d(CS)/dt')
plt.xlabel('CS: scene - face evidence',fontsize=15)
plt.title('Day %i' % d)
plt.colorbar()
plt.show()
#plt.subplot(1,3,3)
plt.figure(figsize=(20,10))
d=2
this_plot_hc = np.nanmean(all_hist[:,:,HC_ind,d],axis=2).T
this_plot_mdd = np.nanmean(all_hist[:,:,MDD_ind,d],axis=2).T
plt.imshow(this_plot_mdd-this_plot_hc,cmap='bwr',vmin=vmin,vmax=vmax,origin='lower')
plt.yticks(np.arange(len(vel_edges))-0.5,labels_vel,fontsize=10)
plt.xticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.ylabel('d(CS)/dt')
plt.xlabel('CS: scene - face evidence',fontsize=15)
plt.title('Day %i' % d)
plt.colorbar()
plt.show()
#plt.show()
# now plot for subject
s = 0
subject_key = 'subject' + str(subjects[s])
fig,ax = plt.subplots()
for d in np.arange(nDays):
    day_key = 'day' + str(d)
    #plt.subplot(1,3,d+1)
#   plt.figure()
    pos = all_x[subject_key][day_key][0]
    vel = all_dx[subject_key][day_key][0]
    #plt.scatter(x,y,alpha=0.5,s = 1.5,color='k')
    sns.jointplot(x=x,y=y,kind='kde',space=0,color='b')
    # pos_range = np.linspace(-1,1,100)
    # vel_range = np.linspace(-2,2,100)
    # posv,velv = np.meshgrid(pos_range,vel_range)
    # pos_deriv = vel[0:-1]
    # vel_deriv = np.diff(vel)
    # plt.quiver(posv,velv,pos_deriv,vel_deriv,alpha=0.75)

    plt.xlim([-1,1])
    plt.ylim([-2,2])
    plt.xlabel('categ sep')
    plt.ylabel('d(categ sep)/dt')
plt.show()

# do 3 separate things for each day
d = 0
mdd_x = []
mdd_dx = []
hc_x = []
hc_dx = []

for s in np.arange(nsubs):
    subject_key = 'subject' + str(subjects[s])
    day_key = 'day' + str(d)
    x = all_x[subject_key][day_key][0]
    y = all_dx[subject_key][day_key][0]
    if subjects[s] < 100:
        hc_x.extend(x)
        hc_dx.extend(y)
    if subjects[s] > 100:
        mdd_x.extend(x.flatten())
        mdd_dx.extend(y)

xnew = hc_x.copy()
xnew.extend(mdd_x)
dxnew = hc_dx.copy()
dxnew.extend(mdd_dx)
data = {}
data['x'] = xnew
data['dx'] = dxnew
subjects = np.ones((len(xnew)))
subjects[0:len(hc_x)] = 0
data['groups'] = subjects
df = pd.DataFrame.from_dict(data)


g = sns.jointplot(x=hc_x,y=hc_dx,kind='kde',space=0,color='k',ratio=3)
#plt.plot([-100,100],[0,0],color='k')
#plt.plot([0,0],[-100,100],color='k')
#g.xlim([-1.5,1.5])
#g.ylim([-1.5,1.5])
plt.show()

g = sns.jointplot(x=mdd_x,y=mdd_dx,kind='kde',space=0,color='r',ratio=3)
#plt.plot([-100,100],[0,0],color='k')
#plt.plot([0,0],[-100,100],color='k')
#g.xlim([-1.5,1.5])
#g.ylim([-1.5,1.5])
plt.show()


i=0
g = sns.JointGrid("x", "dx", df)

for x,group_data in df.groupby("groups"):
    if i==0:
        color = 'k'
        label = 'HC'
    else:
        color = 'r'
        label = 'MDD'
    sns.kdeplot(group_data['x'], ax=g.ax_marg_x, legend=False,color=color)
    sns.kdeplot(group_data['dx'], ax=g.ax_marg_y, vertical=True, legend=False,color=color)
    #g.ax_joint.plot(group_data['x'],group_data['dx'], "o", ms=1,color=color)
    #g.ax_joint.plot(group_data['x'],group_data['dx'], alpha=0.5,color=color)
    #g.ax_joint.scatter(group_data['x'],group_data['dx'],alpha=0.5,color=color)
    #g.ax_joint.kdeplot(group_data['x'],group_data['dx'],alpha=0.5,color=color)
    #g.plot_joint(sns.kdeplot)
    g.plot_joint(sns.distplot,x=group_data['x'],y=group_data['dx'])
    i += 1
plt.show()



mdd_mean = np.mean(neg_dec_day[MDD_ind,:],axis=0)
hc_mean = np.mean(neg_dec_day[HC_ind,:],axis=0)
mdd_err = scipy.stats.sem(neg_dec_day[MDD_ind,:],axis=0)
hc_err = scipy.stats.sem(neg_dec_day[HC_ind,:],axis=0)
alpha=0.3
plt.figure()
for s in np.arange(nsubs):
    if s in HC_ind:
        color = 'k'
    elif s in MDD_ind:
        color = 'r'
    plt.plot(neg_dec_day[s,:],'-',ms=10,color=color,alpha=alpha,lw=2)
plt.errorbar(x=np.arange(nDays),y=hc_mean,yerr=hc_err,color='k',lw=2,label='MDD',fmt='-o',ms=10)
plt.errorbar(x=np.arange(nDays),y=mdd_mean,yerr=mdd_err,color='r',lw=2,label='MDD',fmt='-o',ms=10)
plt.xlabel('day')
plt.ylabel('area under -0.1')
plt.xticks([0,1,2])
plt.show()

# difference between day 1 and day 3?
scipy.stats.ttest_rel(neg_dec_day[MDD_ind,0],neg_dec_day[MDD_ind,2])
scipy.stats.ttest_rel(neg_dec_day[HC_ind,0],neg_dec_day[HC_ind,2])

colors=['k','r']
M = getMADRSscoresALL()
fig = plt.figure(figsize=(10,7))
for s in np.arange(nsubs):
    subjectNum  = subjects[s]
    this_sub_madrs = M[subjectNum]
    madrs_change = this_sub_madrs[1] - this_sub_madrs[0]
    if subjectNum < 100:
        style = 0
    elif subjectNum > 100:
        style = 1
    plt.plot(neg_dec_day[s,2] - neg_dec_day[s,0],madrs_change,marker='.',ms=20,color=colors[style],alpha=0.5)
plt.xlabel('p(decrease|negative) Change 3 - 1')
plt.ylabel('MADRS Change 3 - 1')
plt.show()
#scipy.stats.pearsonr(neg_dec_day[MDD_ind,2] - neg_dec_day[MDD_ind,0])
