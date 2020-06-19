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
from numpy.polynomial.polynomial import polyfit
import matplotlib
import matplotlib.pyplot as plt
font = {'size'   : 22}
import pandas as pd
import seaborn as sns
matplotlib.rc('font', **font)
import csv
from anne_additions.aprime_file import aprime,get_blockType ,get_blockData, get_attCategs, get_imCategs, get_trialTypes_and_responses, get_decMatrices_and_aPrimes
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels
from statsmodels.stats.anova import AnovaRM
from anne_additions.plotting_pretty.commonPlotting import *
import matplotlib.pyplot as plt
import math
from anne_additions.plotting_pretty.thesis_allfunctions import *
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp

# first, define global variables
# define colors
colors = ['#636363','#de2d26'] 
# all subjects and indices for each group
subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,12,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115])
# these are the subjects that got resting state data
func_con_subjects = np.array([3,4,5,6,7,8,9,10,11,12,106,107,108,109,110,111,112,113,114,115])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
nsubs = len(subjects)
# load the MADRS scores from the MADRS.csv file
M = getMADRSscoresALL() 
# get the changes in MADRS scores over time for d1: V1-V5, d2: V1-V6, d3: V1-v7
d1,d2,d3 = getMADRSdiff(M,subjects)
# load arguments about what we wanted to plot
plot_MADRS = int(sys.argv[1]) # where or not to make a MADRS plot
plot_behavGoNoGo = int(sys.argv[2])
plot_gaze = int(sys.argv[3])
plot_transition_matrix = int(sys.argv[4])
plot_connectivity = int(sys.argv[5])
plot_faces = int(sys.argv[6])
plot_networkAnalysis = int(sys.argv[7])
########################################################################################################################
# (1) MADRS depression severity analysis

def categorizeMADRS(M):
  # find the number of subjects in each group by each visit for MDD
  # 9–17= mild, 18–34 = moderate, and > 35 = severe)
  bins = [0,9,18,35,100]
  n_bins = len(bins) - 1
  N_in_each_bin = np.zeros((n_bins,4)) # number in each group for each day
  for s in np.arange(nsubs):
    if s in MDD_ind:
      indices = np.digitize(M[subjects[s]],bins)
      indices_0ind = indices.copy() - 1
      for d in np.arange(4):
        this_score = indices_0ind[d]
        if this_score < n_bins:
          N_in_each_bin[this_score,d] +=1
  return N_in_each_bin

def plotMADRS(M,d1,d2,d3):
  nVisits = 4
  
  # (1) plot MADRS scores over time and calculate statistics
  fig,ax = plt.subplots(figsize=(12,9))
  # plot for each subject
  for s in np.arange(nsubs):
    print(s)
    if subjects[s] < 100:
      style = 0
      plt.plot(np.arange(nVisits),M[subjects[s]],'-',ms=10,color=colors[style],alpha=0.3,lw=2)
    else:
      style = 1
      plt.plot(np.arange(nVisits),M[subjects[s]],'-',ms=10,color=colors[style],alpha=0.3,lw=2)
  HC_scores = np.array([M[subjects[i]] for i in HC_ind])
  MDD_scores = np.array([M[subjects[i]] for i in MDD_ind])
  plt.errorbar(np.arange(nVisits),np.nanmean(HC_scores,axis=0),lw = 5,color=colors[0],yerr=scipy.stats.sem(HC_scores,axis=0,nan_policy='omit'), label='HC',fmt='-o',ms=10)
  plt.errorbar(np.arange(nVisits),np.nanmean(MDD_scores,axis=0),lw = 5,color=colors[1],yerr=scipy.stats.sem(MDD_scores,axis=0,nan_policy='omit'), label='MDD',fmt='-o',ms=10)
  #plt.xticks(np.arange(nVisits),('V1\nPre NF', 'V5\nPost NF', 'V6\n1M FU', 'V7\n3M FU'),fontsize=25)
  plt.xticks(np.arange(nVisits),('','','',''),fontsize=25)
  plt.xlabel('')
  plt.ylabel('') # padding with spaces here so the label is centered on the ticks
  plt.ylabel('MADRS score               ',fontsize=30) # padding with spaces here so the label is centered on the ticks
  plt.ylim([-2,60])
  labels_pos_v = np.arange(0,50,10)
  labels_pos = labels_pos_v.astype(np.str)

  plt.yticks(labels_pos_v,labels_pos,fontsize=25)

  # now add all statistics (MDD group only) and print it so we can check
  x,y=nonNan(MDD_scores[:,0],MDD_scores[:,1])
  t,p = scipy.stats.ttest_rel(x,y)
  addComparisonStat_SYM(p/2,0,1,np.nanmax(MDD_scores),1.5,0.05,'$MDD_1 > MDD_5$')
  printStatsResults('MADRS MDD 1 -> 5',t,p/2,x,y)
  x,y=nonNan(MDD_scores[:,0],MDD_scores[:,2])
  t,p = scipy.stats.ttest_rel(x,y)
  addComparisonStat_SYM(p/2,0,2,np.nanmax(MDD_scores)+7.5,2,0.05,'$MDD_1 > MDD_6$')
  printStatsResults('MADRS MDD 1 -> 6',t,p/2,x,y)
  x,y=nonNan(MDD_scores[:,0],MDD_scores[:,3])
  t,p = scipy.stats.ttest_rel(x,y)
  addComparisonStat_SYM(p/2,0,3,np.nanmax(MDD_scores)+15,2.5,0.05,'$MDD_1 > MDD_7$')
  printStatsResults('MADRS MDD 1 -> 7',t,p/2,x,y)
  sns.despine()
  ax.spines['left'].set_bounds(-2, 40) # this stops the y-axis line from continuing past our ytick labels

  #ax.get_legend().remove()
  plt.savefig('thesis_plots_checked/MADRS.eps')


  ##plt.show()
  return

########################################################################################################################
# (2) - behavioral data from the go/no go task conducted in the CNDS lab (NOT in the scanner)
def plotBehavGoNoGo(subjects):
  all_sadbias, all_happybias, all_neutralface, all_neutralscene, all_happyBlocks, all_sadBlocks = analyzeBehavGoNoGo(subjects)
  #sadbias differences: v5 - v1
  sadbias_diff = all_sadbias[:,2] - all_sadbias[:,0]
  happybias_diff = all_happybias[:,2] - all_happybias[:,0]
  # combine for total a prime score over all conditions:
  combined = np.concatenate((all_neutralscene[:,:,np.newaxis],all_neutralface[:,:,np.newaxis],all_happyBlocks[:,:,np.newaxis],all_sadBlocks[:,:,np.newaxis]),axis=2)
  combined_avg = np.nanmean(combined,axis=2)
  combined_diff = combined_avg[:,2] - combined_avg[:,0]
  combined_neutral = np.concatenate((all_neutralscene[:,:,np.newaxis],all_neutralface[:,:,np.newaxis]),axis=2)
  combined_neutral_avg = np.nanmean(combined_neutral,axis=2)
  combined_diff_neutral = combined_neutral_avg[:,2] - combined_neutral_avg[:,0]
  sadBlocks_diff = all_sadBlocks[:,2] - all_sadBlocks[:,0]
  topL=0.3 # this is where to put the stats

  # (1) plot difference in sad bias between groups
  # the sad bias = aprime neutral - aprime sad 
  # thus, if neg faces are MORE distracting than neutral faces, there would be 
  # a positive sad bias because they would do better (large A prime) for neutral blocks
  yfont=0.4
  yfontsize=25
  stat = all_sadbias
  fig,ax = plotPosterStyle_DF(stat,subjects)
  plt.ylim([-.5,0.5])
  plt.text(-.4,yfont,'harder to ignore\nnegative faces',fontsize=yfontsize,va='top',ha='left')
  plt.text(-.4,-1*yfont,'harder to ignore\nneutral faces',fontsize=yfontsize,va='bottom', ha='left')
  plt.xticks(np.arange(4),('V1\nPre NF', 'V3\nMid NF', 'V5\nPost NF', 'V6\n1M FU'),fontsize=30)
  plt.plot([-1,5],[0,0], '--', color='k', lw=1)
  x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p,-.2,.2,topL,0.05,0,r'$MDD \neq HC$')
  printStatsResults('behav sad bias 2-tailed MDD/HC, V1',t,p,x,y)
  x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
  t,p = scipy.stats.ttest_rel(x,y)
  addComparisonStat_SYM(p/2,0.2,2.2,topL+.2,0.05,0,'$MDD_1 < MDD_5$')
  printStatsResults('behav sad bias 1-tailed MDD only V1-V5, V1',t,p/2,x,y)
  # plt.ylabel("A' negative bias",fontsize=30)
  # plt.title('Negative bias', fontsize=32)
  # plt.xticks(np.arange(4),('V1\nPre NF', 'V3\nMid NF', 'V5\nPost NF', 'V6\n1M FU'),fontsize=30)
  plt.ylabel('')
  plt.title('')
  plt.xticks(np.arange(4),('', '', '', ''),fontsize=30)
  plt.xlabel("")
  plt.savefig('thesis_plots_checked/aprime_sadbias.eps')
  #plt.show()

  # (2) plot difference in happy bias between groups
  # the happy bias aprime neutral - aprime happy
  # so if happy faces were harder to attend to, there would be
  # a positive happy bias because they would do better (large A prime) for neutral blocks
  stat = all_happybias
  fig,ax = plotPosterStyle_DF(stat,subjects)
  plt.ylim([-.5,0.5])
  plt.text(-.4,yfont,'harder to attend\nto positive faces',fontsize=yfontsize,va='top',ha='left')
  plt.text(-.4,-1*yfont,'harder to attend\nto neutral faces',fontsize=yfontsize,va='bottom', ha='left')
  plt.plot([-1,5],[0,0], '--', color='k', lw=1)
  x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p,-.2,.2,topL,0.05,0, r'$MDD \neq HC$')
  printStatsResults('behav happy bias 2-tailed MDD/HC, V1',t,p,x,y)
  x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
  t,p = scipy.stats.ttest_rel(x,y)
  addComparisonStat_SYM(p/2,0.2,2.2,topL+.2,0.05,0,'$MDD_1 < MDD_5$')
  printStatsResults('behav happy bias 1-tailed MDD only V1-V5, V1',t,p/2,x,y)
  # plt.ylabel("A' anti-positive bias", fontsize=30)
  # plt.title('Anti-positive bias', fontsize=32)
  # plt.xticks(np.arange(4),('V1\nPre NF', 'V3\nMid NF', 'V5\nPost NF', 'V6\n1M FU'),fontsize=30)
  plt.ylabel('')
  plt.title('')
  plt.xticks(np.arange(4),('', '', '', ''),fontsize=30)
  plt.xlabel("")
  plt.savefig('thesis_plots_checked/aprime_positivebias.eps')
  #plt.show()
  
  return combined_diff

########################################################################################################################
# (2) - Eye tracking Data
def plotGaze(subjects,M,d1,d2,d3):
  # get gaze data from .mat file 
  first_orientation,total_viewing_time,fixation_durations_first_time = getGazeData(subjects)
  # all of these matrices are in the shape: (n_subjects,n_trials,n_days,n_emotions)

  # say which emotion corresponds to each index
  emo = ['DYS', 'THR', 'NEU', 'POS']
  DYSPHORIC = 1
  THREAT = 2
  NEUTRAL = 3
  POSITIVE = 4

  # first we look at FIRST ORIENTATION data -
  # this is the first image on which a subject fixates in a given trial
  # it is returned as a ratio of probability for each category for a given day

  # (1) plot the group difference for first orientation on dysphoric images
  stat = np.nanmean(first_orientation[:,:,:,DYSPHORIC-1],axis=1)
  topL=0.3
  fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
  plt.ylim([0,.65])
  #plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))
  plt.yticks(np.array([0,0.2,0.4,0.6]),fontsize=25)
  x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,-.2,.2,0.4,0.05,0,'$MDD > HC$')
  printStatsResults('first orientation dysphoric 1-tailed MDD/HC V1',t,p/2,x,y)

  x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,0.8,1.2,topL,0.05,0,'$MDD > HC$')
  printStatsResults('first orientation dysphoric 1-tailed MDD/HC V3',t,p/2,x,y)

  x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,1.8,2.2,topL,0.05,0,'$MDD > HC$')
  printStatsResults('first orientation dysphoric 1-tailed MDD only V1-V5',t,p/2,x,y)

  x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
  t,p = scipy.stats.ttest_rel(x,y)
  addComparisonStat_SYM(p/2,0.2,2.2,topL+.2,0.05,0,'$MDD_1 > MDD_5$')
  printStatsResults('first orientation dysphoric 1-tailed MDD only V1-V5',t,p/2,x,y)

  x,y = nonNan(stat[MDD_ind,3],stat[HC_ind,3])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,2.8,3.2,topL,0.05,0,'$MDD > HC$')
  printStatsResults('first orientation dysphoric 1-tailed MDD-HC V6',t,p/2,x,y)
  # plt.ylabel('ratio orientation - dysphoric', fontsize=30)
  # plt.title('Initial orientation to dysphoric images', fontsize=32)
  # plt.xticks(np.arange(4),('V1\nPre NF', 'V3\nMid NF', 'V5\nPost NF', 'V6\n1M FU'),fontsize=30)
  plt.ylabel('')
  plt.title('')
  plt.xticks(np.arange(4),('', '', '', ''),fontsize=30)
  plt.xlabel('')
  plt.savefig('thesis_plots_checked/gaze_orientation_dysphoric.eps')
  #plt.show()

  # (2) plot the group difference for first orientation on happy images
  stat = np.nanmean(first_orientation[:,:,:,POSITIVE-1],axis=1)
  topL=0.3
  fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
  plt.ylim([0,.65])
  plt.yticks(np.array([0,0.2,0.4,0.6]),fontsize=25)
  #plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))
  x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,-.2,.2,0.4,0.05,0,'$MDD < HC$')
  printStatsResults('first orientation happy 1-tailed MDD-HC V1',t,p/2,x,y)
  x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p,0.8,1.2,topL,0.05,0,r'$MDD \neq HC$')
  printStatsResults('first orientation happy 1-tailed MDD-HC V3',t,p/2,x,y)
  x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p,1.8,2.2,topL,0.05,0,r'$MDD \neq HC$')
  printStatsResults('first orientation happy 1-tailed MDD-HC V5',t,p/2,x,y)
  x,y = nonNan(stat[MDD_ind,3],stat[HC_ind,3])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p,2.8,3.2,topL,0.05,0,r'$MDD \neq HC$')
  printStatsResults('first orientation happy 1-tailed MDD-HC V6',t,p/2,x,y)
  # plt.ylabel('ratio orientation - positive', fontsize=30)
  # plt.title('Initial orientation to positive images', fontsize=32)
  # plt.xticks(np.arange(4),('V1\nPre NF', 'V3\nMid NF', 'V5\nPost NF', 'V6\n1M FU'),fontsize=30)
  plt.ylabel('')
  plt.title('')
  plt.xticks(np.arange(4),('', '', '', ''),fontsize=30)
  plt.xlabel('')
  plt.savefig('thesis_plots_checked/gaze_orientation_positive.eps')
  #plt.show()

  # next we look at TOTAL VIEWING RATIO -
  # this is the proportion of time subjects viewed each category,
  # normalized by the total number of fixations for that trial

  # (3) - plot dysphoric total viewing ratio
  stat = np.nanmean(total_viewing_time[:,:,:,DYSPHORIC-1],axis=1)
  topL=0.33
  fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
  plt.ylim([0,.62])
  plt.yticks(np.array([0,0.2,0.4,0.6]),fontsize=25)
  x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,-.2,.2,0.4,0.025,0,'$MDD > HC$')
  printStatsResults('total viewing ratio dysphoric 1-tailed MDD-HC V1',t,p/2,x,y)
  x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,0.8,1.2,topL,0.025,0,'$MDD > HC$')
  printStatsResults('total viewing ratio dysphoric 1-tailed MDD-HC V3',t,p/2,x,y)

  x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,1.8,2.2,topL,0.025,0,'$MDD > HC$')
  printStatsResults('total viewing ratio dysphoric 1-tailed MDD-HC V5',t,p/2,x,y)

  x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
  t,p = scipy.stats.ttest_rel(x,y)
  addComparisonStat_SYM(p/2,0.2,2.2,0.51,0.025,0,'$MDD_1 > MDD_5$')
  printStatsResults('total viewing ratio dysphoric 1-tailed MDD V1-5',t,p/2,x,y)

  x,y = nonNan(stat[MDD_ind,3],stat[HC_ind,3])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,2.8,3.2,topL,0.025,0,'$MDD > HC$')
  printStatsResults('total viewing ratio dysphoric 1-tailed MDD-HC V6',t,p/2,x,y)
  # plt.ylabel('ratio total viewing - dysphoric', fontsize=30)
  # plt.title('Total viewing of dysphoric images', fontsize=32)
  # plt.xticks(np.arange(4),('V1\nPre NF', 'V3\nMid NF', 'V5\nPost NF', 'V6\n1M FU'),fontsize=30)
  plt.ylabel('')
  plt.title('')
  plt.xticks(np.arange(4),('', '', '', ''),fontsize=30)
  plt.xlabel('')
  plt.savefig('thesis_plots_checked/gaze_totalviewing_dysphoric.eps')
  #plt.show()

  # (4) plot total view time ratio to dysphoric and MADRS decrease relationship
  data_mat = np.nanmean(total_viewing_time[:,:,:,DYSPHORIC-1],axis=1) # average over all trials
  # calculate changes in dysphoric viewing: V5 - V1
  all_neg_change = data_mat[:,2] - data_mat[:,0]

  fig,ax = plt.subplots(figsize=(19,12))
  sns.despine()
  for s in np.arange(nsubs):
  #for s in keep:
    subjectNum  = subjects[s]
    if subjectNum < 100:
      style = 0
    elif subjectNum > 100:
      style = 1
    plt.plot(-1*all_neg_change[s],-1*d1[s],marker='.',ms=30,color=colors[style])
  plt.xlabel('improvement in dysphoric viewing time',fontsize=20)
  plt.ylabel('improvement in depression severity',fontsize=20)
  plt.xticks(fontsize=30)
  plt.yticks(fontsize=30)
  #plt.xlim([-.5,.8])
  plt.title('MADRS vs. dysphoric viewing time change')
  x,y = nonNan(-1*all_neg_change,-1*d1)
  r,p=scipy.stats.pearsonr(x,y)
  text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(-.2,20,text, ha='left',va='top',color='k',fontsize=25)
  printStatsResults('linear corr: dysphoric total time viewing and MADRS',r,p,x,y)
  x,y = nonNan(-1*all_neg_change[MDD_ind],-1*d1[MDD_ind])
  b, m = polyfit(x, y, 1)
  plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
  r,p=scipy.stats.pearsonr(x,y)
  text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(-.2,15,text, ha='left',va='top',color='k',fontsize=25)
  printStatsResults('linear corr: dysphoric total time viewing and MADRS - MDD only',r,p,x,y)
  plt.savefig('thesis_plots_checked/total_viewing_vs_MADRS.pdf')
  #plt.show()

  # calculate interaction between group and change in total viewing time to dysphoric
  stat = np.nanmean(total_viewing_time[:,:,:,DYSPHORIC-1],axis=1)
  df = convertMatToDF(stat,subjects)
  df2 = df[(df['day']==0) | (df['day'] == 2)]
  model = ols('data ~ group*day',df2).fit()
  print('interaction - total viewing dysphoric * group')
  print(model.summary())

  # new 6/4/20: check interaction with a t-test
  all_diff = stat[:,2] - stat[:,0]
  # now positive = reductions
  x,y = nonNan(-1*all_diff[MDD_ind],-1*all_diff[HC_ind])
  t,p = scipy.stats.ttest_ind(x,y)
  printStatsResults('t-test interaction unpaired',t,p/2,x,y)

  # next we look at FIRST FIXATION MAINTENANCE -
  # this is the proportion of time subjects viewed each category,
  # for the first time ONLY that they maintained fixation on that image
  # for the given trial, normalized by the total time they viewed images in the trial
  
  # (5) - plot dysphoric fixation time
  stat = np.nanmean(fixation_durations_first_time[:,:,:,DYSPHORIC-1],axis=1)
  topL=0.27
  fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
  plt.ylim([0,.42])
  plt.yticks(np.array([0,0.2,0.4]),fontsize=25)
  #plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))
  x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,-.2,.2,topL,0.01,0,'$MDD > HC$')
  printStatsResults('fixation ratio dysphoric 1-tailed MDD-HC V1',t,p/2,x,y)

  x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,0.8,1.2,topL,0.01,0,'$MDD > HC$')
  printStatsResults('fixation ratio dysphoric 1-tailed MDD-HC V3',t,p/2,x,y)

  x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,1.8,2.2,topL,0.01,0,'$MDD > HC$')
  printStatsResults('fixation ratio dysphoric 1-tailed MDD-HC V5',t,p/2,x,y)

  x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
  t,p = scipy.stats.ttest_rel(x,y)
  addComparisonStat_SYM(p/2,0.2,2.2,0.35,0.01,0,'$MDD_1 > MDD_5$')
  printStatsResults('fixation ratio dysphoric 1-tailed MDD V1-V5',t,p/2,x,y)

  x,y = nonNan(stat[MDD_ind,3],stat[HC_ind,3])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,2.8,3.2,topL,0.01,0,'$MDD > HC$')
  printStatsResults('fixation ratio dysphoric 1-tailed MDD-HC V6',t,p/2,x,y)
  # plt.xticks(np.arange(4),('V1\nPre NF', 'V3\nMid NF', 'V5\nPost NF', 'V6\n1M FU'),fontsize=30)
  # plt.ylabel('ratio maintenance - dysphoric', fontsize=30)
  # plt.title('Maintenance on dysphoric images', fontsize=32)
  plt.ylabel('')
  plt.title('')
  plt.xticks(np.arange(4),('', '', '', ''),fontsize=30)
  plt.xlabel('')
  plt.savefig('thesis_plots_checked/gaze_firstfix_dysphoric.eps')
  #plt.show()

  # calculate interaction between changes in fixation during to dysphoric and group
  stat = np.nanmean(fixation_durations_first_time[:,:,:,DYSPHORIC-1],axis=1)
  df = convertMatToDF(stat,subjects)
  df2 = df[(df['day']==0) | (df['day'] == 2)]
  model = ols('data ~ group*day',df2).fit()
  model.summary()
  print('interaction - fixation dysphoric * group')
  print(model.summary())

  # new 6/4/20: check interaction with a t-test
  all_diff = stat[:,2] - stat[:,0]
  # now positive = reductions
  x,y = nonNan(-1*all_diff[MDD_ind],-1*all_diff[HC_ind])
  t,p = scipy.stats.ttest_ind(x,y)
  printStatsResults('t-test interaction',t,p/2,x,y)

  # (6) plot relationship between changes in fixation maintenance ratio dysphoric and MADRS scores
  data_mat =  np.nanmean(fixation_durations_first_time[:,:,:,DYSPHORIC-1],axis=1)
  all_neg_change = data_mat[:,2] - data_mat[:,0] # change from V5 - V1

  fig,ax = plt.subplots(figsize=(19,12))
  sns.despine()
  for s in np.arange(nsubs):
    subjectNum  = subjects[s]
    if subjectNum < 100:
      style = 0
    elif subjectNum > 100:
      style = 1
    plt.plot(-1*all_neg_change[s],-1*d1[s],marker='.',ms=30,color=colors[style])
  plt.xlabel('improvement in dysphoric fixation',fontsize=20)
  plt.ylabel('improvement in depression severity',fontsize=20)
  plt.xticks(fontsize=30)
  plt.yticks(fontsize=30)
  plt.xlim([-.1,.2])
  plt.title('MADRS vs. dysphoric maintenance Change')
  x,y = nonNan(-1*all_neg_change,-1*d1)
  r,p=scipy.stats.pearsonr(x,y)
  text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(-.27,20,text, ha='left',va='top',color='k',fontsize=25)
  printStatsResults('linear corr: dysphoric fixation and MADRS',r,p,x,y)
  x,y = nonNan(-1*all_neg_change[MDD_ind],-1*d1[MDD_ind])
  b, m = polyfit(x, y, 1)
  plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
  r,p=scipy.stats.pearsonr(x,y)
  text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
  printStatsResults('linear corr: dysphoric fixation and MADRS - MDD only',r,p,x,y)

  plt.text(-.27,15,text, ha='left',va='top',color='k',fontsize=25)
  plt.savefig('thesis_plots_checked/first_fix_vs_MADRS.pdf')
  #plt.show()
  return



######################################################### TRANSITION MATRIX ANALYSIS ######################################
def plotTransitionMatrix(subjects,M,d1,d2,d3):
  # build subject matrix containing all neurofeedback data
  master_x,all_x = buildMasterDict(subjects,1) # the 1 here means only take the first half, middle half, and last half of days 1/2/3
  # specify custom bin sizes
  bins = [-1.   , -0.975, -0.9, -0.8 ,-0.7,-0.55,-0.4,-0.2,0,0.2,0.4,0.55,0.7, 0.8 ,  0.9 , 0.975, 1. ]
  nbins=len(bins)

  # (1) plot histogram of all values for time spent in each bin
  fig,ax = plt.subplots(figsize=(24,10))
  sns.despine()
  counts, bins = np.histogram(master_x,bins)
  total_val = np.sum(counts)
  counts_norm = counts/total_val
  plt.hist(bins[:-1], bins, weights=counts_norm,color='k',alpha=0.25,edgecolor='k',linewidth=3)
  # labels for the bins
  labels_pos = np.array(bins).astype(np.float)
  labels_pos = np.around(labels_pos,decimals=2).astype(np.str)
  labels_pos[0] = '-1'
  labels_pos[-1] = '1'
  labels_pos[8] = '0'
  for l in np.arange(len(labels_pos)):
    # alternate every other label spacing for x tick labels
    if np.mod(l,2):
      # changing here on 5/17 to not put every other label
      #labels_pos[l] = '\n' + labels_pos[l]
      labels_pos[l] = ' '
  plt.xticks(bins,labels_pos,fontsize=40)
  plt.yticks(np.array([0.025,0.1]),['2.5', '10'],fontsize=45)
  #plt.ylabel('frequency (%)', fontsize=45)
  plt.ylabel('')
  plt.xlim([-1,1])
  plt.ylim([0.02,.1])
  ax.tick_params(axis='y', which='major', pad=10)
  ax.tick_params(axis='x', which='major', pad=0)
  #plt.xlabel('attention state', fontsize=45)
  plt.xlabel('')
  #plt.title('Scene - face classification distribution', fontsize=32)
  plt.savefig('thesis_plots_checked/paper_nf_histogram.eps')
  #plt.show()

  # next, get transition matrix data 
  # the 2 and 3 are the two different TR shift settings saying to go
  # 2 and 3 TR shifts ahead for "stickiness" and then average total for one result
  all_matrices,p_state,subject_averages = buildTransitionMatrix(subjects,all_x,bins,2,3)
  nDays=3
  # get diagonal data - meaning the conditional probability to stay in the same state
  diagonal_data = np.zeros((nsubs,nbins-1,nDays))
  for s in np.arange(nsubs):
      for d in np.arange(nDays):
          ex = all_matrices[:,:,s,d]
          diagonal = np.diagonal(ex)
          diagonal_data[s,:,d] = diagonal
  # find largest group difference on each day
  diag_diff = np.abs(np.nanmean(diagonal_data[MDD_ind,:,0],axis=0) - np.nanmean(diagonal_data[HC_ind,:,0],axis=0))
  print('early NF largest group difference is')
  print(np.argmax(diag_diff))
  diag_diff = np.abs(np.nanmean(diagonal_data[MDD_ind,:,2],axis=0) - np.nanmean(diagonal_data[HC_ind,:,2],axis=0))
  np.argmax(diag_diff)
  print('late NF largest group difference is')
  print(np.argmax(diag_diff))

  # (2) plot diagonal data for early and late neurofeedback with subplots
  # make label positions all on one line again
  labels_pos = np.array(bins).astype(np.float)
  labels_pos_bin_loc = np.arange(nbins)-0.5
  labels_pos_half_bin_loc = labels_pos_bin_loc[::2]
  labels_pos_half = labels_pos[::2] # take every other
  labels_pos_half_str = np.around(labels_pos_half,decimals=2).astype(np.str)
  labels_pos_half_str[0] = '-1'
  labels_pos_half_str[-1] = '1'
  labels_pos_half_str[4] = '0'
  fig, ax = plotPosterStyle_multiplePTS(diagonal_data[:,:,np.array([0,2])],subjects,0)
 
  plt.subplot(1,2,1)
  plt.ylim([0,.55])
  plt.yticks(np.linspace(0,.5,3),fontsize=25)
  plt.xticks(labels_pos_half_bin_loc,labels_pos_half_str,fontsize=20)
  # instead, let's plot every other bin to save space
  ax[0].tick_params(axis='x', which='major', pad=10)
  # plt.xlabel('attention state at t', fontsize=35)
  # plt.ylabel('sustained attention\nat t + 5 s', fontsize=40)
  plt.xlabel('')
  plt.ylabel('')
  # plt.title('Early NF', fontsize=40)
  plt.title('')
  x,y = nonNan(diagonal_data[MDD_ind,0,0],diagonal_data[HC_ind,0,0])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,0,0,0.5,0.01,0,'')
  plt.subplot(1,2,2)
  ax[1].tick_params(axis='x', which='major', pad=10)
  plt.ylim([0,.55])
  plt.yticks(np.linspace(0,.5,3),fontsize=25)
  plt.xticks(labels_pos_half_bin_loc,labels_pos_half_str,fontsize=20)
  x,y = nonNan(diagonal_data[MDD_ind,15,2],diagonal_data[HC_ind,15,2])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,0,0,0.5,0.01,0,'')
  #plt.legend()
  # plt.xlabel('attention state at t', fontsize=35)
  plt.xlabel('')
  #ax.tick_params(axis='x', which='major', pad=10)
  # plt.title('Late NF', fontsize=40)
  plt.title('')
  plt.savefig('thesis_plots_checked/paper_nf_diagonal.eps')
  #plt.show()


  # (3) plot the change specifically in negative stickiness from early to late NF
  i=0
  j=0
  stat = all_matrices[i,j,:,:]
  fig,ax = plotPosterStyle_DF(stat[:,np.array([0,2])],subjects) # <-- the [0,2] grabs just day 1 and 3 of NF
  x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,-.2,.2,0.73,0.05,0,'$MDD > HC$')
  printStatsResults('transition matrix neg stickiness: 1-tailed MDD HC early NF ',t,p/2,x,y)
  x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0)
  printStatsResults('transition matrix neg stickiness: 1-tailed MDD HC late NF ',t,p/2,x,y)
  x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
  t,p = scipy.stats.ttest_rel(x,y)
  addComparisonStat_SYM(p/2,0.2,1.2,np.nanmax(stat)+.1,0.05,0,'$MDD_E > MDD_L$')
  printStatsResults('transition matrix neg stickiness: 1-tailed MDD early-late NF ',t,p/2,x,y)
  # add interaction - I hardcoded this one after calculating it as 0.045 above
  addComparisonStat_SYM(0.045,0,1,np.nanmax(stat)+.30,0,0,'group:visit',0)
  # plt.ylabel('sustained negative attention',fontsize=30)
  # plt.title('Sustained negative attention',fontsize=32)
  # plt.xticks(np.arange(2),('Early NF','Late NF'),fontsize=30)
  plt.ylabel('')
  plt.title('')
  plt.xticks(np.arange(2),('',''),fontsize=30)
  plt.ylim([0,1.25])
  labels_pos_v = np.array([0,0.5,1])
  labels_pos = labels_pos_v.astype(np.str)
  plt.yticks(labels_pos_v,labels_pos,fontsize=25)
  plt.xlabel('')
  plt.savefig('thesis_plots_checked/nf_saddest_bargraph.eps')
  #plt.show()

  # calculate the interaction between negative stickiness and group scores over time
  i=0
  j=0
  stat = all_matrices[i,j,:,:]
  df = convertMatToDF(stat,subjects)
  model = ols('data ~ group*day',df).fit()
  print('interaction - negative stickiness group*day')
  print(model.summary())

  # compute interactio n differently
  all_diff = stat[:,2] - stat[:,0]
  # multiply by negative 1 so it's in units of reduction
  HC_diff = -1*all_diff[HC_ind]
  MDD_diff = -1*all_diff[MDD_ind]
  x,y = nonNan(MDD_diff,HC_diff)
  t,p = scipy.stats.ttest_ind(x,y)
  printStatsResults('diff in changes over time ',t,p/2,x,y)

  # (4) plot the change specifically in neutral correct sustained attention from early to late NF
  i=15
  j=15
  stat = all_matrices[i,j,:,:]
  fig,ax = plotPosterStyle_DF(stat[:,np.array([0,2])],subjects)
  x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD < HC$' )
  printStatsResults('transition matrix correct attention: 1-tailed MDD HC early NF ',t,p/2,x,y)
  x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0,'$MDD < HC$' )
  printStatsResults('transition matrix correct attention: 1-tailed MDD HC late NF ',t,p/2,x,y)
  x,y = nonNan(stat[HC_ind,0],stat[HC_ind,2])
  t,p = scipy.stats.ttest_rel(x,y)
  addComparisonStat_SYM(p/2,-0.2,0.8,np.nanmax(stat)+0.2,0.05,0,'$HC_E < HC_L$')
  printStatsResults('transition matrix correct attention: 1-tailed HC early-late NF ',t,p/2,x,y)
  plt.ylabel('sustained correct attention',fontsize=30)
  plt.title('Sustained correct attention',fontsize=32)

  plt.ylim([0,1.25])
  plt.xticks(np.arange(2),('Early NF','Late NF'),fontsize=30)
  labels_pos_v = np.array([0,0.5,1])
  labels_pos = labels_pos_v.astype(np.str)
  plt.yticks(labels_pos_v,labels_pos,fontsize=25)
  plt.xlabel('')
  plt.savefig('thesis_plots_checked/nf_correct_bargraph.pdf')
  #plt.show()
  x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
  t,p = scipy.stats.ttest_rel(x,y)
  printStatsResults('transition matrix correct attention: 1-tailed MDD early-late NF ',t,p/2,x,y)

  # calculate interactions with correct sustained attention between group and day
  i=15
  j=15
  stat = all_matrices[i,j,:,:]
  df = convertMatToDF(stat,subjects)
  model = ols('data ~ group*day',df).fit()
  print('interaction - sustained attention interaction group*day')
  print(model.summary())

  # (5) plot linear relationship between changes in negative stickiness and MADRS scores
  # changed 6/3/20 to only plot MDD group
  data_mat = all_matrices[0,0,:,:]
  all_neg_change = data_mat[:,2] - data_mat[:,0]

  #fig,ax = plt.subplots(figsize=(19,12))
  fig,ax = plt.subplots(figsize=(12,9))
  sns.despine()
  for s in np.arange(nsubs):
    subjectNum  = subjects[s]
    if subjectNum < 100:
      style = 0
    elif subjectNum > 100:
      style = 1
      plt.plot(-1*all_neg_change[s],-1*d1[s],marker='.',ms=30,color=colors[style])

  #plt.title('MADRS vs. NF Change')
  # x,y = nonNan(-1*all_neg_change,-1*d1)
  # r,p=scipy.stats.pearsonr(x,y)
  # text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
  # plt.text(-.47,20,text, ha='left',va='top',color='k',fontsize=22)
  # printStatsResults('linear neg stickiness - MADRS change ',r,p,x,y)
  x,y = nonNan(-1*all_neg_change[HC_ind],-1*d1[HC_ind])
  r,p=scipy.stats.pearsonr(x,y)
  printStatsResults('linear neg stickiness - MADRS change - HC only',r,p,x,y)

  x,y = nonNan(-1*all_neg_change[MDD_ind],-1*d1[MDD_ind])
  b, m = polyfit(x, y, 1)
  plt.plot(x, b + m * x, '-',alpha=0.6,lw=5, color='k')
  r,p=scipy.stats.pearsonr(x,y)
  text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(-.47,15,text, ha='left',va='top',color='k',fontsize=22)
  printStatsResults('linear neg stickiness - MADRS change - MDD only',r,p,x,y)

  labels_pos_v = np.array([-0.5,-0.25,0,0.25,0.5,0.75])
  labels_pos = labels_pos_v.astype(np.str)
  plt.xticks(labels_pos_v,labels_pos,fontsize=20)
  plt.yticks(np.arange(-10,30,10), fontsize=20)
  # plt.title('Depression severity and sustained\nnegative attention relationship', fontsize=25)
  # plt.xlabel('reduction in sustained negative attention',fontsize=25)
  # plt.ylabel('reduction in MADRS score',fontsize=25)
  plt.title('')
  plt.xlabel('')
  plt.ylabel('')
  plt.ylim([-10,22])
  # add a little space to the axis tick labels so they don't bump into each other
  #ax.tick_params(axis='both', which='major', pad=15)
  ax.tick_params(axis='x', which='major', pad=10)
  plt.savefig('thesis_plots_checked/NF_vs_MADRS.eps')
  #plt.show()

  # (6) plot average classification results for groups 
  stat = subject_averages
  fig,ax = plotPosterStyle_DF(stat[:,np.array([0,2])],subjects) # early and late NF only
  x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD_1 < HC_1$')
  printStatsResults('average categ sep - 2-tailed MDD/HC early NF',t,p,x,y)
  x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0)
  printStatsResults('average categ sep - 2-tailed MDD/HC late NF',t,p,x,y)
  # x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
  # t,p = scipy.stats.ttest_rel(x,y)
  # addComparisonStat_SYM(p/2,0.2,1.2,np.nanmax(stat)+.2,0.05,0,'$MDD_1 > MDD_3$')
  
  plt.ylim([-.5,.5])
  labels_pos_v = np.array([-.5,0,0.5])
  labels_pos = labels_pos_v.astype(np.str)
  plt.yticks(labels_pos_v,labels_pos,fontsize=25)
  # plt.xticks(np.arange(2),('Early NF','Late NF'),fontsize=30)
  # plt.ylabel('mean(scene - face classification)',fontsize=30)
  # plt.title('Average classification', fontsize=30)
  plt.xticks(np.arange(2),('',''),fontsize=30)
  plt.ylabel('')
  plt.title('')
  plt.xlabel('')
  plt.savefig('thesis_plots_checked/nf_clf_avg.eps')
  #plt.show()

  # do both groups decrease significantly?
  x,y = nonNan(stat[:,0],stat[:,2])
  t,p = scipy.stats.ttest_rel(x,y)
  printStatsResults('average categ sep - 2-tailed combined groups - do they go down over time',t,p,x,y)

  return

######################################################### RESTING STATE ANALYSIS - NODE ######################################
def plotRestingState(subjects,func_con_subjects, M, d1, d2, d3):
  HC_ind_con = np.argwhere(func_con_subjects<100)[:,0]
  MDD_ind_con = np.argwhere(func_con_subjects>100)[:,0]
  average_within_mat, amyg_con = getFunctionalCon(func_con_subjects)

  # (1) plot default mode network connectivity - network 0
  dmn_connectivity = average_within_mat[0,0,:,:] # subjects x visit
  stat = dmn_connectivity
  topL=0.2
  fig,ax = plotPosterStyle_DF(stat,func_con_subjects)
  plt.ylim([0,.35])
  plt.xticks(np.arange(2),('V2\nPre NF', 'V4\nPost NF'),fontsize=30)
  plt.yticks(np.arange(0,0.4,0.1),fontsize=25)
  x,y = nonNan(stat[MDD_ind_con,0],stat[HC_ind_con,0])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,-.2,.2,topL,0.01,0,'$MDD > HC$')
  printStatsResults('DMN connectivity- 1-tailed MDD HC group difference, pre NF',t,p/2,x,y)
  x,y = nonNan(stat[MDD_ind_con,1],stat[HC_ind_con,1])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,0.8,1.2,topL,0.01,0,'$MDD > HC$')
  printStatsResults('DMN connectivity- 1-tailed MDD HC group difference, post NF',t,p/2,x,y)
  x,y = nonNan(stat[MDD_ind_con,0],stat[MDD_ind_con,1])
  t,p = scipy.stats.ttest_rel(x,y)
  addComparisonStat_SYM(p/2,0.2,1.2,topL+.1,0.007,0,'$MDD_2 > MDD_4$')
  printStatsResults('DMN connectivity- 1-sided MDD pre-post',t,p/2,x,y)
  plt.ylabel('DMN within-network connectivity',fontsize=30)
  plt.title('DMN connectivity', fontsize=32)
  plt.xlabel('')
  plt.savefig('thesis_plots_checked/con_dmn.pdf')
  #plt.show()

  # now calculate the change in DMN connectivity to compare to MADRS improvement
  data_mat =dmn_connectivity
  all_neg_change1 = data_mat[:,1] - data_mat[:,0]
  all_subjects_func_con = np.zeros((nsubs,)) * np.nan
  # this will make it so all subjects with connectivity scores will have a value
  # and all subjects without connectivity with have nan (won't be plotted)
  for s in np.arange(nsubs):
    if subjects[s] in func_con_subjects:
      ind_f = np.argwhere(subjects[s] == func_con_subjects)[0][0]
      all_subjects_func_con[s] = all_neg_change1[ind_f]
  all_neg_change = all_subjects_func_con # so this is in units of negative = DMN connectivity goes down

  # (2) plot linear DMN changes and MADRS changes
  fig,ax = plt.subplots(figsize=(19,12))
  sns.despine()
  for s in np.arange(nsubs):
  #for s in keep:
    subjectNum  = subjects[s]
    if subjectNum < 100:
      style = 0
    elif subjectNum > 100:
      style = 1
    plt.plot(-1*all_neg_change[s],-1*d1[s],marker='.',ms=30,color=colors[style])
  plt.xlabel('improvement in DMN connectivity',fontsize=20)
  plt.ylabel('improvement in depression severity',fontsize=20)
  plt.xticks(fontsize=30)
  plt.yticks(fontsize=30)
  #plt.xlim([-.5,.8])
  plt.title('MADRS vs. NF Change')
  x,y = nonNan(-1*all_neg_change,-1*d1)
  r,p=scipy.stats.pearsonr(x,y)
  text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(-.47,20,text, ha='left',va='top',color='k',fontsize=25)
  printStatsResults('linear relationship: DMN and MADRS changes',r,p,x,y)
  x,y = nonNan(-1*all_neg_change[MDD_ind],-1*d1[MDD_ind])
  # mutliply by 1 --> pos = reduction of DMN connectivity [good]
  # because we're comparing -1 * MDD score changes so positive is also [good]
  # --> we expect positive relationship bc DMN within-network connectivity is bad
  b, m = polyfit(x, y, 1)
  plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
  r,p=scipy.stats.pearsonr(x,y)
  text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(-.47,15,text, ha='left',va='top',color='k',fontsize=25)
  printStatsResults('linear relationship: DMN and MADRS changes - MDD only',r,p,x,y)
  plt.savefig('thesis_plots_checked/DMN_vs_MADRS.pdf')
  #plt.show()

  # (3) plot frontoparietal connectivity - node index # 1
  fpn_connectivity = average_within_mat[1,1,:,:] # subjects x visit
  stat = fpn_connectivity
  topL=0.2
  fig,ax = plotPosterStyle_DF(stat,func_con_subjects)
  plt.ylim([0,.35])
  plt.yticks(np.arange(0,0.4,0.1),fontsize=25)
  plt.xticks(np.arange(2),('V2\nPre NF', 'V4\nPost NF'),fontsize=30)
  x,y = nonNan(stat[MDD_ind_con,0],stat[HC_ind_con,0])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p,-.2,.2,topL,0.05,0,r'$MDD \neq HC$')
  printStatsResults('FPN connectivity 2-tailed MDD-HC, pre NF',t,p,x,y)
  x,y = nonNan(stat[MDD_ind_con,1],stat[HC_ind_con,1])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p,0.8,1.2,topL,0.05,0,r'$MDD \neq HC$')
  printStatsResults('FPN connectivity 2-tailed MDD-HC, post NF',t,p,x,y)
  x,y = nonNan(stat[MDD_ind_con,0],stat[MDD_ind_con,1])
  t,p = scipy.stats.ttest_rel(x,y)
  printStatsResults('FPN connectivity 1-tailed change MDD pre to post NF',t,p/2,x,y)
  # x,y = nonNan(stat[MDD_ind_con,0],stat[MDD_ind_con,1])
  # t,p = scipy.stats.ttest_rel(x,y)
  # addComparisonStat_SYM(p/2,0.2,1.2,topL+.3,0.05,0,'$MDD_1 > MDD_5$')
  plt.ylabel('FPN within-network connectivity',fontsize=30)
  plt.title('FPN connectivity', fontsize=32)
  plt.xlabel('')
  plt.savefig('thesis_plots_checked/con_fpn.pdf')
  #plt.show()

  # (4) plot DMN to FPN connectivity
  dmn_to_fpn = average_within_mat[1,0,:,:]
  stat = dmn_to_fpn
  topL=0.00057
  fig,ax = plotPosterStyle_DF(stat,func_con_subjects)
  plt.ylim([-.0004,.0007])
  plt.xticks(np.arange(2),('V2\nPre NF', 'V4\nPost NF'),fontsize=30)
  plt.yticks(fontsize=25)
  x,y = nonNan(stat[MDD_ind_con,0],stat[HC_ind_con,0])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p,-.2,.2,topL,0.000005,0,r'$MDD \neq HC$')
  printStatsResults('DMN-FPN connectivity 2-tailed MDD-HC, pre NF',t,p,x,y)
  x,y = nonNan(stat[MDD_ind_con,1],stat[HC_ind_con,1])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p,0.8,1.2,topL,0.000005,0,r'$MDD \neq HC$')
  printStatsResults('DMN-FPN connectivity 2-tailed MDD-HC, post NF',t,p,x,y)

  # x,y = nonNan(stat[MDD_ind_con,0],stat[MDD_ind_con,1])
  # t,p = scipy.stats.ttest_rel(x,y)
  # addComparisonStat_SYM(p/2,0.2,1.2,topL+.3,0.05,0,'$MDD_1 > MDD_5$')
  plt.ylabel('DMN - FPN connectivity',fontsize=30)
  plt.title('DMN - FPN connectivity', fontsize=32)

  plt.xlabel('')
  ax.ticklabel_format(axis='y', style='scientific',scilimits=(-2,2))
  plt.savefig('thesis_plots_checked/con_dmn_fpn.pdf')
  #plt.show()


  # check interaction between group and day dmn to fpn interaction
  stat = dmn_to_fpn
  df = convertMatToDF(stat,func_con_subjects)
  model = ols('data ~ group*day',df).fit()
  print('DMN-FPN connectivity interaction: group*day')
  print(model.summary())


  ####################################################### RESTING STATE ANALYSIS - ROI ######################################
  # (5) plots amygdala - FPN connectivity
  stat = amyg_con
  topL=np.nanmax(stat)+ 0.05
  fig,ax = plotPosterStyle_DF(stat,func_con_subjects)
  plt.ylim([-.25,.25])
  plt.yticks(np.arange(-.2,.3,.1),fontsize=25)
  plt.xticks(np.arange(2),('V2\nPre NF', 'V4\nPost NF'),fontsize=30)
  x,y = nonNan(stat[MDD_ind_con,0],stat[HC_ind_con,0])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p,-.2,.2,topL,0.005,0,r'$MDD \neq HC$')
  printStatsResults('LA-FPN connectivity 2-tailed MDD-HC, pre NF',t,p,x,y)
  x,y = nonNan(stat[MDD_ind_con,1],stat[HC_ind_con,1])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p,0.8,1.2,topL,0.005,0,r'$MDD \neq HC$')
  printStatsResults('LA-FPN connectivity 2-tailed MDD-HC, post NF',t,p,x,y)
  # x,y = nonNan(stat[MDD_ind_con,0],stat[MDD_ind_con,1])
  # t,p = scipy.stats.ttest_rel(x,y)
  # addComparisonStat_SYM(p/2,0.2,1.2,topL+.3,0.05,0,'$MDD_1 > MDD_5$')
  plt.ylabel('FPN - LA connectivity',fontsize=30)
  plt.title('FPN - LA connectivity', fontsize=32)

  plt.xlabel('')
  plt.savefig('thesis_plots_checked/con_fpn_amyg.pdf')
  #plt.show()

  return


####################################################### FACES TASK ANALYSIS ############################################
def plotFaces(subjects):
  colors_dark = ['#636363','#de2d26']
  # second input is ROI - this amygdala overlapping
  negative_ts, neutral_ts, happy_ts, nTR = getFaces3dTProjectData(subjects,'amyg_overlapping')
  # subtract negative - neutral time series
  negative_diff = negative_ts - neutral_ts
  # test interaction betweem group and day on index 14
  stat = negative_diff[:,14,:]
  df = convertMatToDF(stat,subjects)
  model = ols('data ~ group*day',df).fit()
  print('interaction faces time series neg-neutral group*day')
  print(model.summary())

  # new 6/4/20: check interaction with a t-test
  all_diff = stat[:,1] - stat[:,0]
  print('shape of stat matrix is')
  print(np.shape(stat))
  # now positive = reductions
  x,y = nonNan(-1*all_diff[MDD_ind],-1*all_diff[HC_ind])
  t,p = scipy.stats.ttest_ind(x,y)
  printStatsResults('t-test interaction',t,p/2,x,y)

  # show that the average time series does NOT differ by group
  # for the average block activity we want TRs 7:16 
  # (because 5 is the block start + shift by 2)
  # for time only during the block - UNSHIFTED
  negative_diff_block = negative_diff[:,7:16,:]
  day = 0
  # average time series over block, now shifted
  mdd_average = np.mean(negative_diff_block[MDD_ind,:,day],axis=1)
  hc_average = np.mean(negative_diff_block[HC_ind,:,day],axis=1)
  t,p = scipy.stats.ttest_ind(mdd_average,hc_average)
  printStatsResults('average neg-neutral time series on day 1 difference, 1-sided',t,p/2,mdd_average,hc_average)

  # (1) plot difference in neg - neutral faces pre-neurofeedback
  fig = plt.subplots(figsize=(19,10))
  sns.despine()
  x = np.arange(nTR)
  day=0
  y = negative_diff[HC_ind,:,day]
  ym = np.mean(y,axis=0)
  yerr = scipy.stats.sem(y,axis=0)
  plt.errorbar(x,ym,yerr=yerr,linewidth=5,color=colors_dark[0])
  y = negative_diff[MDD_ind,:,day]
  ym = np.mean(y,axis=0)
  yerr = scipy.stats.sem(y,axis=0)
  plt.errorbar(x,ym,yerr=yerr,linewidth=5,color=colors_dark[1])
  labels_pos_v = np.concatenate([np.arange(-5,nTR)])
  labels_pos = labels_pos_v.astype(np.str)
  plt.xticks(np.arange(nTR),labels_pos,fontsize=25)
  plt.yticks(fontsize=25)
  plt.text(5, 0.35, 'stim on', ha='center', va='bottom', color='k',fontsize=25)
  plt.text(13, 0.35, 'stim off', ha='center', va='bottom', color='k',fontsize=25)
  # plt.ylabel('negative - neutral LA activity', fontsize=30)
  # plt.xlabel('TR relative to block start', fontsize=30)
  # plt.title('Pre NF', fontsize=35)
  plt.ylabel('')
  plt.xlabel('')
  plt.title('')
  plt.plot([5,5],[-10,0.3],'--', lw=2, c='k')
  plt.plot([13,13],[-10,0.3],'--', lw=2, c='k')
  plt.ylim([-.7,.7])
  x,y = nonNan(negative_diff[MDD_ind,14,day],negative_diff[HC_ind,14,day])
  t,p = scipy.stats.ttest_ind(x,y)
  #plt.legend(loc=2)
  addComparisonStat_SYM(p/2,14,14,0.4,0.05,0,'$MDD > HC$')
  printStatsResults('neg-neutral time series on day 1, tr 14, MDD-HC 1-sided',t,p/2,x,y)
  plt.savefig('thesis_plots_checked/faces_LA_diff_day_1.eps')
  #plt.show()

  # (2) plot difference in neg - neutral faces post-neurofeedback
  fig = plt.subplots(figsize=(19,10))
  sns.despine()
  x = np.arange(nTR)
  day=1
  y = negative_diff[HC_ind,:,day]
  ym = np.mean(y,axis=0)
  yerr = scipy.stats.sem(y,axis=0)
  plt.errorbar(x,ym,yerr=yerr,linewidth=5,color=colors_dark[0])
  y = negative_diff[MDD_ind,:,day]
  ym = np.mean(y,axis=0)
  yerr = scipy.stats.sem(y,axis=0)
  plt.errorbar(x,ym,yerr=yerr,linewidth=5,color=colors_dark[1])
  labels_pos_v = np.concatenate([np.arange(-5,nTR)])
  labels_pos = labels_pos_v.astype(np.str)
  plt.xticks(np.arange(nTR),labels_pos,fontsize=25)
  plt.yticks(fontsize=25)
  plt.text(5, 0.35, 'stim on', ha='center', va='bottom', color='k',fontsize=25)
  plt.text(13, 0.35, 'stim off', ha='center', va='bottom', color='k',fontsize=25)
  # plt.ylabel('negative - neutral LA activity', fontsize=30)
  # plt.xlabel('TR relative to block start', fontsize=30)
  # plt.title('Post NF', fontsize=35)
  plt.ylabel('')
  plt.xlabel('')
  plt.title('')
  plt.plot([5,5],[-10,0.3],'--', lw=2, c='k')
  plt.plot([13,13],[-10,0.3],'--', lw=2, c='k')
  plt.ylim([-.7,.7])
  x,y = nonNan(negative_diff[MDD_ind,14,day],negative_diff[HC_ind,14,day])
  t,p = scipy.stats.ttest_ind(x,y)
  addComparisonStat_SYM(p/2,14,14,0.4,0.05,0,'$MDD > HC$')
  plt.legend(loc=2)
  plt.savefig('thesis_plots_checked/faces_LA_diff_day_3.eps')
  #plt.show()

  x,y = nonNan(negative_diff[MDD_ind,14,1],negative_diff[HC_ind,14,1])
  t,p = scipy.stats.ttest_ind(x,y)
  printStatsResults('neg-neutral time series on day 3, tr 14, MDD-HC 1-sided',t,p/2,x,y)

  x,y = nonNan(negative_diff[MDD_ind,14,0],negative_diff[MDD_ind,14,1])
  t,p = scipy.stats.ttest_rel(x,y)
  printStatsResults('neg-neutral time series day1-day 3, tr 14, MDD 1-sided',t,p/2,x,y)

  x,y = nonNan(negative_diff[HC_ind,14,0],negative_diff[HC_ind,14,1])
  t,p = scipy.stats.ttest_rel(x,y)
  printStatsResults('neg-neutral time series day1-day 3, tr 14, HC 1-sided',t,p/2,x,y)

  # (3) plot linear relationship between decrease in amyg activity in this TR and MADRS scores
  data_mat =  negative_diff[:,14,:]
  all_neg_change = data_mat[:,1] - data_mat[:,0] # post - pre neurofeedback
  fig,ax = plt.subplots(figsize=(19,12))
  sns.despine()
  for s in np.arange(nsubs):
    subjectNum  = subjects[s]
    if subjectNum < 100:
      style = 0
    elif subjectNum > 100:
      style = 1
    plt.plot(-1*all_neg_change[s],-1*d1[s],marker='.',ms=30,color=colors[style])
  plt.xlabel('improvement in LA response',fontsize=20)
  plt.ylabel('improvement in depression severity',fontsize=20)
  plt.xticks(fontsize=30)
  plt.yticks(fontsize=30)
  #plt.xlim([-.1,.2])
  plt.title('MADRS vs. LA response Change')
  x,y = nonNan(-1*all_neg_change,-1*d1)
  r,p=scipy.stats.pearsonr(x,y)
  text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(-.27,20,text, ha='left',va='top',color='k',fontsize=25)
  printStatsResults('linear relationship: LA activity dec, MADRS',r,p,x,y)
  x,y = nonNan(-1*all_neg_change[MDD_ind],-1*d1[MDD_ind])
  b, m = polyfit(x, y, 1)
  plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
  r,p=scipy.stats.pearsonr(x,y)
  text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(-.27,15,text, ha='left',va='top',color='k',fontsize=25)
  printStatsResults('linear relationship: LA activity dec, MADRS - MDD only',r,p,x,y)
  plt.savefig('thesis_plots_checked/LA_change_vs_MADRS.pdf')
  #plt.show()

  # not used - plotting each day at a time
  # fig = plt.subplots(figsize=(19,10))
  # sns.despine()
  # x = np.arange(nTR)
  # day=0
  # y = negative_ts[HC_ind,:,day]
  # ym = np.mean(y,axis=0)
  # yerr = scipy.stats.sem(y,axis=0)
  # plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[0], label='HC negative')
  # y = neutral_ts[HC_ind,:,day]
  # ym = np.mean(y,axis=0)
  # yerr = scipy.stats.sem(y,axis=0)
  # plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC neutral')
  # y = negative_ts[MDD_ind,:,day]
  # ym = np.mean(y,axis=0)
  # yerr = scipy.stats.sem(y,axis=0)
  # plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[1], label='MDD negative')
  # y = neutral_ts[MDD_ind,:,day]
  # ym = np.mean(y,axis=0)
  # yerr = scipy.stats.sem(y,axis=0)
  # plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD neutral')
  # labels_pos_v = np.concatenate([np.arange(-5,nTR)])
  # labels_pos = labels_pos_v.astype(np.str)
  # plt.xticks(np.arange(nTR),labels_pos,fontsize=30)
  # plt.text(5, 0.5, 'stim on', ha='center', va='bottom', color='k',fontsize=25)
  # plt.text(13, 0.5, 'stim off', ha='center', va='bottom', color='k',fontsize=25)
  # plt.ylabel('z-scored activity')
  # plt.xlabel('TR relative to block start')
  # plt.plot([5,5],[-10,10],'--', lw=1, c='k')
  # plt.plot([13,13],[-10,10],'--', lw=1, c='k')
  # plt.ylim([-.7,.7])
  # plt.legend()
  # plt.savefig('thesis_plots_checked/faces_LA_day_1.pdf')
  # #plt.show()

  # day=1
  # fig = plt.subplots(figsize=(19,10))
  # sns.despine()
  # y = negative_ts[HC_ind,:,day]
  # ym = np.mean(y,axis=0)
  # yerr = scipy.stats.sem(y,axis=0)
  # plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[0], label='HC negative')
  # y = neutral_ts[HC_ind,:,day]
  # ym = np.mean(y,axis=0)
  # yerr = scipy.stats.sem(y,axis=0)
  # plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC neutral')
  # y = negative_ts[MDD_ind,:,day]
  # ym = np.mean(y,axis=0)
  # yerr = scipy.stats.sem(y,axis=0)
  # plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[1], label='MDD negative')
  # y = neutral_ts[MDD_ind,:,day]
  # ym = np.mean(y,axis=0)
  # yerr = scipy.stats.sem(y,axis=0)
  # plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD neutral')
  # labels_pos_v = np.concatenate([np.arange(-5,nTR)])
  # labels_pos = labels_pos_v.astype(np.str)
  # plt.xticks(np.arange(nTR),labels_pos,fontsize=30)
  # plt.text(5, 0.5, 'stim on', ha='center', va='bottom', color='k',fontsize=25)
  # plt.text(13, 0.5, 'stim off', ha='center', va='bottom', color='k',fontsize=25)
  # plt.ylabel('z-scored activity')
  # plt.xlabel('TR relative to block start')
  # plt.plot([5,5],[-10,10],'--', lw=1, c='k')
  # plt.plot([13,13],[-10,10],'--', lw=1, c='k')
  # plt.ylim([-.7,.7])
  # plt.legend()
  # plt.savefig('thesis_plots_checked/faces_LA_day_3.pdf')
  # #plt.show()

  # now look at if you can let time vary there's an interaction - build data frame
  # negative_diff_initial = negative_diff_block[:,:,0]
  # negative_diff_final = negative_diff_block[:,:,1]

  # data = negative_diff_initial.flatten()
  # data2 = negative_diff_final.flatten()
  # both_data = np.concatenate((data,data2),axis=0)
  # nTR = 9
  # # goes through all subjects first
  # subjects = np.repeat(np.arange(nsubs),nTR)
  # subjects2 = np.concatenate((subjects,subjects),axis=0)
  # TR = np.tile(np.arange(nTR),nsubs)
  # TR2 = np.concatenate((TR,TR),axis=0)

  # group = [''] * len(subjects)
  # for i in np.arange(len(subjects)):
  #   if subjects[i] in MDD_ind:
  #     group[i] = 'MDD'
  #   elif subjects[i] in HC_ind:
  #     group[i] = 'HC'
  # group2 = group + group
  # day = np.repeat(np.arange(2),len(subjects))
  # all_data = {}
  # all_data['activity'] = data
  # all_data['TR'] = TR
  # all_data['group'] = group
  # all_data['subjects'] = subjects
  # df = pd.DataFrame.from_dict(all_data)
  # model = ols('activity ~ group*TR',df).fit()
  # model.summary()
  # all_data2 = {}
  # all_data2['activity'] = both_data
  # all_data2['TR'] = TR2
  # all_data2['group'] = group2
  # all_data2['subjects'] = subjects2
  # all_data2['day'] = day
  # df2 = pd.DataFrame.from_dict(all_data2)
  # model = ols('activity ~ group*day',df2).fit()
  # model.summary()
  # # difference between group
  # df3 = df2[df2['TR']==7]
  # model = ols('activity ~ group*day',df3).fit()
  # model.summary()

  return


########################################################################################################################
# correlation network analyses
def plotNetworkAnalysis(subjects,combined_diff):
  # load in correlation results - the 0 means we want to look at all neurofeedback runs
  perception_correlation, attention_correlation, perception_attention_correlation = correlateROICS(subjects,0)
  perception_run_avg = np.nanmean(perception_correlation,axis=1)
  perception_avg = np.nanmean(perception_run_avg,axis=1)
  attention_run_avg = np.nanmean(attention_correlation,axis=1)
  attention_avg = np.nanmean(attention_run_avg,axis=1)


  ### first repeat Megan's analysis:
  # correlate average contribution of each network and total behavior change in aprime
  # so this is ALL NEUROFEEDBACK RUNS!! correlation with ALL behavior

  # (1) plot linear relationship for each network and behavior improvements
  text1_x = 0.8
  text1_y = -.08
  spacing = 0.07
  fig,ax = plt.subplots(figsize=(20,10))
  plt.subplot(1,2,1)
  x = perception_avg
  y = combined_diff
  sns.despine()
  for s in np.arange(nsubs):
    subjectNum  = subjects[s]
    if subjectNum < 100:
      style = 0
    elif subjectNum > 100:
      style = 1
    plt.plot(x[s],y[s],marker='.',ms=30,color=colors[style],alpha=0.9)
  plt.xlabel('wb - network correlation',fontsize=30)
  plt.ylabel("A' improvement",fontsize=30)
  plt.title('Perceptual network', fontsize=35)
  plt.xticks(np.arange(0.6,1,.1),fontsize=25)
  plt.xlim([0.55,0.9])
  plt.yticks(np.arange(-.2,.3,.1), fontsize=25)
  plt.ylim([-.2,.2])
  r,p=scipy.stats.pearsonr(x,y)
  text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: avg perception network, avg behav change',r,p,x,y)
  x1,y1 = nonNan(x[MDD_ind],y[MDD_ind])
  r,p=scipy.stats.pearsonr(x1,y1)
  text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y - spacing,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: avg perception network, avg behav change - MDD only',r,p,x1,y1)
  plt.subplot(1,2,2)
  x = attention_avg
  sns.despine()
  for s in np.arange(nsubs):
    subjectNum  = subjects[s]
    if subjectNum < 100:
      style = 0
    elif subjectNum > 100:
      style = 1
    plt.plot(x[s],y[s],marker='.',ms=30,color=colors[style],alpha=0.9)
  plt.xlabel('wb - network correlation',fontsize=30)
  plt.ylabel('')
  plt.xticks(fontsize=25)
  plt.title('Attentional network', fontsize=35)
  plt.xlim([0.45,0.9])
  plt.ylim([-.2,.2])
  plt.yticks(np.arange(-.2,.3,.1), fontsize=25)
  r,p=scipy.stats.pearsonr(x,y)
  text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: avg attention network, avg behav change',r,p,x,y)
  x1,y1 = nonNan(x[MDD_ind],y[MDD_ind])
  r,p=scipy.stats.pearsonr(x1,y1)
  text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y - spacing,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: avg attention network, avg behav change - MDD only',r,p,x1,y1)
  plt.savefig('thesis_plots_checked/correlation_networks_original_way.pdf')
  #plt.show()


  #### second - NF IMPROVEMENT - take half this time
  # this time, look at the changes in network contribution and compare to other measures
  # the 1 in the function means to take half of the runs on neurofeedback days instead of all runs
  perception_correlationH, attention_correlationH, perception_attention_correlationH = correlateROICS(subjects,1)
  perception_run_avgH = np.nanmean(perception_correlationH,axis=1)
  perception_change = perception_run_avgH[:,2] - perception_run_avgH[:,0]
  attention_run_avgH = np.nanmean(attention_correlationH,axis=1)
  attention_change = attention_run_avgH[:,2] - attention_run_avgH[:,0]
  # get transition matrix data to compare improvements
  master_x,all_x = buildMasterDict(subjects,1)
  bins = [-1.   , -0.975, -0.9, -0.8 ,-0.7,-0.55,-0.4,-0.2,0,0.2,0.4,0.55,0.7, 0.8 ,  0.9 , 0.975, 1. ]
  all_matrices,p_state,subject_averages = buildTransitionMatrix(subjects,all_x,bins,2,3)
  # get difference post - pre
  neg_stickiness = all_matrices[0,0,:,2] - all_matrices[0,0,:,0]
  correct_stickiness = all_matrices[15,15,:,2] - all_matrices[15,15,:,0]

  # (2) plot linear relationship for network changes and behavior changes
  text1_x = -0.2
  text1_y = 0.18
  spacing = 0.07
  fig,ax = plt.subplots(figsize=(20,10))
  plt.subplot(1,2,1)
  x = perception_change
  y = combined_diff
  sns.despine()
  for s in np.arange(nsubs):
    subjectNum  = subjects[s]
    if subjectNum < 100:
      style = 0
    elif subjectNum > 100:
      style = 1
    plt.plot(x[s],y[s],marker='.',ms=30,color=colors[style],alpha=0.9)
  plt.xlabel(r'$\Delta$wb - network correlation',fontsize=30)
  plt.ylabel("A' improvement",fontsize=30)
  plt.title('Perceptual network', fontsize=35)
  plt.xticks(np.arange(-.2,.3,.1),fontsize=25)
  plt.xlim([-0.23,0.23])
  plt.yticks(np.arange(-.2,.3,.1), fontsize=25)
  plt.ylim([-.2,.2])
  r,p=scipy.stats.pearsonr(x,y)
  text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: perception network change, avg behav change',r,p,x,y)
  r,p=scipy.stats.pearsonr(x[MDD_ind],y[MDD_ind])
  text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y-spacing,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: perception network change, avg behav change - MDD only',r,p,x[MDD_ind],y[MDD_ind])
  plt.subplot(1,2,2)
  x = attention_change
  sns.despine()
  for s in np.arange(nsubs):
    subjectNum  = subjects[s]
    if subjectNum < 100:
      style = 0
    elif subjectNum > 100:
      style = 1
    plt.plot(x[s],y[s],marker='.',ms=30,color=colors[style],alpha=0.9)
  plt.title('Attentional network', fontsize=35)
  plt.xlabel(r'$\Delta$wb - network correlation',fontsize=30)
  plt.ylabel('')
  plt.xticks(np.arange(-.2,.6,.1),fontsize=25)
  plt.xlim([-0.23,0.55])
  plt.yticks(np.arange(-.2,.3,.1), fontsize=25)
  plt.ylim([-.2,.2])
  r,p=scipy.stats.pearsonr(x,y)
  text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: attention network change, avg behav change',r,p,x,y)
  r,p=scipy.stats.pearsonr(x[MDD_ind],y[MDD_ind])
  text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y-spacing,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: attention network change, avg behav change - MDD only',r,p,x[MDD_ind],y[MDD_ind])
  plt.savefig('thesis_plots_checked/correlation_networks_change_corr.pdf')
  #plt.show()

  # (3) plot network contribution change and change in negative stickiness
  text1_x = 0.15
  text1_y = 0.65
  spacing = 0.2
  fig,ax = plt.subplots(figsize=(20,10))
  plt.subplot(1,2,1)
  x = perception_change
  y = -1*neg_stickiness
  sns.despine()
  for s in np.arange(nsubs):
    subjectNum  = subjects[s]
    if subjectNum < 100:
      style = 0
    elif subjectNum > 100:
      style = 1
    plt.plot(x[s],y[s],marker='.',ms=30,color=colors[style],alpha=0.9)
  plt.xlabel(r'$\Delta$wb - network correlation',fontsize=30)
  plt.ylabel('reduction in sustained negative attention',fontsize=30)
  plt.xticks(np.arange(-.2,.3,.1),fontsize=25)
  plt.title('Perceptual network', fontsize=35)
  plt.xlim([-0.23,0.23])
  plt.yticks(fontsize=25)
  #plt.xlim([0.45,0.9])
  #plt.ylim([-.2,.25])
  x1,y1 = nonNan(x,y)
  r,p=scipy.stats.pearsonr(x1,y1)
  text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: perception network change, neg stickiness',r,p,x1,y1)
  x1,y1 = nonNan(x[MDD_ind],y[MDD_ind])
  r,p=scipy.stats.pearsonr(x1,y1)
  text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y-spacing,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: perception network change, neg stickiness - MDD only',r,p,x1,y1)
  
  plt.subplot(1,2,2)
  x = attention_change
  text1_x = 0.37
  sns.despine()
  for s in np.arange(nsubs):
    subjectNum  = subjects[s]
    if subjectNum < 100:
      style = 0
    elif subjectNum > 100:
      style = 1
    plt.plot(x[s],y[s],marker='.',ms=30,color=colors[style],alpha=0.9)
  plt.xlabel(r'$\Delta$wb - network correlation',fontsize=30)
  plt.ylabel('')
  plt.xticks(np.arange(-.2,.6,.1),fontsize=25)
  plt.xlim([-0.23,0.55])
  plt.yticks(fontsize=25)
  plt.title('Attentional network', fontsize=35)
  #plt.xlim([0.45,0.9])
  #plt.ylim([-.2,.25])
  x1,y1 = nonNan(x,y)
  r,p=scipy.stats.pearsonr(x1,y1)
  text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: attention network change, neg stickiness',r,p,x1,y1)
  x1,y1 = nonNan(x[MDD_ind],y[MDD_ind])
  r,p=scipy.stats.pearsonr(x1,y1)
  text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y-spacing,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: attention network change, neg stickiness - MDD only',r,p,x1,y1)
  plt.savefig('thesis_plots_checked/correlation_networks_change_corr_stickiness.pdf')
  #plt.show()

  # (4) plot network contribution changes and correct sustained attention changes
  text1_x = -0.2
  text1_y = 0.57
  spacing = 0.2
  fig,ax = plt.subplots(figsize=(20,10))
  plt.subplot(1,2,1)
  x = perception_change
  y = correct_stickiness
  sns.despine()
  for s in np.arange(nsubs):
    subjectNum  = subjects[s]
    if subjectNum < 100:
      style = 0
    elif subjectNum > 100:
      style = 1
    plt.plot(x[s],y[s],marker='.',ms=30,color=colors[style],alpha=0.9)
  plt.xlabel(r'$\Delta$wb - network correlation',fontsize=30)
  plt.ylabel('improvement in sustained correct attention',fontsize=30)
  plt.title('Perceptual network', fontsize=35)
  plt.xticks(np.arange(-.2,.3,.1),fontsize=25)
  plt.xlim([-0.23,0.23])
  plt.yticks(fontsize=25)
  #plt.xlim([0.45,0.9])
  #plt.ylim([-.2,.25])
  x1,y1 = nonNan(x,y)
  r,p=scipy.stats.pearsonr(x1,y1)
  text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: perception network change, correct attention',r,p,x1,y1)
  x1,y1 = nonNan(x[MDD_ind],y[MDD_ind])
  r,p=scipy.stats.pearsonr(x1,y1)
  text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y-spacing,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: perception network change, correct attention - MDD only',r,p,x1,y1)
  plt.subplot(1,2,2)
  x = attention_change
  sns.despine()
  for s in np.arange(nsubs):
    subjectNum  = subjects[s]
    if subjectNum < 100:
      style = 0
    elif subjectNum > 100:
      style = 1
    plt.plot(x[s],y[s],marker='.',ms=30,color=colors[style],alpha=0.9)
  plt.xlabel(r'$\Delta$wb - network correlation',fontsize=30)
  plt.ylabel('')
  plt.xticks(np.arange(-.2,.6,.1),fontsize=25)
  plt.title('Attentional network', fontsize=35)
  plt.xlim([-0.23,0.55])
  plt.yticks(fontsize=25)
  #plt.xlim([0.45,0.9])
  #plt.ylim([-.2,.25])
  x1,y1 = nonNan(x,y)
  r,p=scipy.stats.pearsonr(x1,y1)
  text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: attention network change, correct attention',r,p,x1,y1)
  x1,y1 = nonNan(x[MDD_ind],y[MDD_ind])
  r,p=scipy.stats.pearsonr(x1,y1)
  text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
  plt.text(text1_x,text1_y-spacing,text, ha='left',va='top',color='k',fontsize=20)
  printStatsResults('linear relationship: attention network change, correct attention - MDD only',r,p,x1,y1)
  plt.savefig('thesis_plots_checked/correlation_networks_change_corr_correct_attention.pdf')
  #plt.show()


  # not used - plot linear relationship between network changes and sad bias blocks specifically
  # fig,ax = plt.subplots(figsize=(20,10))
  # plt.subplot(1,2,1)
  # x = perception_change
  # y = sadBlocks_diff 
  # sns.despine()
  # for s in np.arange(nsubs):
  #   subjectNum  = subjects[s]
  #   if subjectNum < 100:
  #     style = 0
  #   elif subjectNum > 100:
  #     style = 1
  #   plt.plot(x[s],y[s],marker='.',ms=30,color=colors[style],alpha=0.9)
  # plt.xlabel(r'$\Delta$correlation: wb - perceptual network',fontsize=20)
  # plt.ylabel(r'behavioral training $\Delta$A',fontsize=20)
  # plt.xticks(fontsize=15)
  # plt.yticks(fontsize=20)
  # #plt.xlim([0.45,0.9])
  # plt.ylim([-.2,.25])
  # r,p=scipy.stats.pearsonr(x,y)
  # text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
  # plt.text(-.2,.21,text, ha='left',va='top',color='k',fontsize=20)
  # r,p=scipy.stats.pearsonr(x[MDD_ind],y[MDD_ind])
  # text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
  # plt.text(-.2,.15,text, ha='left',va='top',color='k',fontsize=20)

  # plt.subplot(1,2,2)
  # x = attention_change
  # sns.despine()
  # for s in np.arange(nsubs):
  #   subjectNum  = subjects[s]
  #   if subjectNum < 100:
  #     style = 0
  #   elif subjectNum > 100:
  #     style = 1
  #   plt.plot(x[s],y[s],marker='.',ms=30,color=colors[style],alpha=0.9)
  # plt.xlabel(r'$\Delta$correlation: wb - attentional network',fontsize=20)
  # plt.ylabel('')
  # plt.xticks(fontsize=15)
  # #plt.xlim([0.45,0.9])
  # plt.ylim([-.2,.25])
  # r,p=scipy.stats.pearsonr(x,y)
  # text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
  # plt.text(-.2,.21,text, ha='left',va='top',color='k',fontsize=20)#plt.ylim([-.2,.2])
  # r,p=scipy.stats.pearsonr(x[MDD_ind],y[MDD_ind])
  # text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
  # plt.text(-.2,.15,text, ha='left',va='top',color='k',fontsize=20)
  # #plt.title('Transitions %i shifted ahead' % nshift1)
  # plt.savefig('thesis_plots_checked/correlation_networks_change_corr_distract_sad.pdf')
  # #plt.show()


  return
##########################################################################################

def main():
  """now we run the functions based on user input"""
  if plot_MADRS:
    plotMADRS(M,d1,d2,d3)
  if plot_behavGoNoGo:
    combined_diff = plotBehavGoNoGo(subjects)
  if plot_gaze:
    plotGaze(subjects,M,d1,d2,d3)
  if plot_transition_matrix:
    plotTransitionMatrix(subjects,M,d1,d2,d3)
  if plot_connectivity:
    plotRestingState(subjects,func_con_subjects, M, d1, d2, d3)
  if plot_faces:
    plotFaces(subjects)
  if plot_networkAnalysis:
    combined_diff = plotBehavGoNoGo(subjects)
    plotNetworkAnalysis(subjects,combined_diff)

  return

if __name__ == "__main__":
    # execute only if run as a script
    main()
