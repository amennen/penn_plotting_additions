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
font = {'weight' : 'normal',
        'size'   : 22}
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

# define colors
colors = ['#636363','#de2d26']
subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,12,101,102,103,104,105,106, 107,108,109,110,111,112,113,114,115])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
nsubs = len(subjects)
M = getMADRSscoresALL()
d1,d2,d3 = getMADRSdiff(M,subjects)
plot_MADRS = sys.argv[1] # where or not to make a MADRS plot
########################################################################################################################
# (1) MADRS SCORES analysis

def plotMADRS(M,d1,d2,d3):
  nVisits = 4
  # plot MADRS scores over time and calculate statistics
  fig,ax = plt.subplots(figsize=(12,9))
  sns.despine()
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
  plt.xticks(np.arange(nVisits),('pre NF', 'post NF', '1M FU', '3M FU'),fontsize=30)
  plt.xlabel('Visit',fontsize=35)
  plt.ylabel('MADRS score',fontsize=40)
  plt.ylim([-2,60])
  labels_pos_v = np.arange(0,50,10)
  labels_pos = labels_pos_v.astype(np.str)
  plt.yticks(labels_pos_v,labels_pos,fontsize=30)
  x,y=nonNan(MDD_scores[:,0],MDD_scores[:,1])
  t,p = scipy.stats.ttest_rel(x,y)
  addComparisonStat_SYM(p/2,0,1,np.nanmax(MDD_scores),1.5,0.05,'$MDD_1 > MDD_5$')
  printStatsResults('MADRS MDD 1 -> 5',t,p/2)
  x,y=nonNan(MDD_scores[:,0],MDD_scores[:,2])
  t,p = scipy.stats.ttest_rel(x,y)
  addComparisonStat_SYM(p/2,0,2,np.nanmax(MDD_scores)+7.5,2,0.05,'$MDD_1 > MDD_6$')
  printStatsResults('MADRS MDD 1 -> 6',t,p/2)
  x,y=nonNan(MDD_scores[:,0],MDD_scores[:,3])
  t,p = scipy.stats.ttest_rel(x,y)
  addComparisonStat_SYM(p/2,0,3,np.nanmax(MDD_scores)+15,2.5,0.05,'$MDD_1 > MDD_7$')
  printStatsResults('MADRS MDD 1 -> 7',t,p/2)
  #ax.get_legend().remove()
  plt.savefig('thesis_plots/MADRS.pdf')
  #plt.show()
  return

########################################################################################################################
# FIRST - get all behavioral scores
def plotBehavGoNoGo(subjects):
all_sadbias, all_happybias, all_neutralface, all_neutralscene, all_happyBlocks, all_sadBlocks = analyzeBehavGoNoGo(subjects)
#sadbias differences: v5 - v1
sadbias_diff = all_sadbias[:,2] - all_sadbias[:,0]
happybias_diff = all_happybias[:,2] - all_happybias[:,0]
# combine for total a prime score:
combined = np.concatenate((all_neutralscene[:,:,np.newaxis],all_neutralface[:,:,np.newaxis],all_happyBlocks[:,:,np.newaxis],all_sadBlocks[:,:,np.newaxis]),axis=2)
combined_avg = np.nanmean(combined,axis=2)
combined_diff = combined_avg[:,2] - combined_avg[:,0]
combined_neutral = np.concatenate((all_neutralscene[:,:,np.newaxis],all_neutralface[:,:,np.newaxis]),axis=2)
combined_neutral_avg = np.nanmean(combined_neutral,axis=2)
combined_diff_neutral = combined_neutral_avg[:,2] - combined_neutral_avg[:,0]

sadBlocks_diff = all_sadBlocks[:,2] - all_sadBlocks[:,0]

# (1) differences in sad bias? - aprime neutral - aprime sad (if neg faces distracting would be positive sad bias)
topL=0.3

stat = all_sadbias
fig,ax = plotPosterStyle_DF(stat,subjects)
plt.ylim([-.6,0.6])
plt.text(-.4,.6,'neg faces are \nmore distracting',fontsize=10,va='top',ha='left')
plt.text(-.4,-.6,'neutral faces are \nmore distracting',fontsize=10,va='bottom', ha='left')
plt.xticks(np.arange(4),('Pre NF', 'Mid NF', 'Post NF', '1M FU'))
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,topL,0.05,0,'$MDD > HC$')
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,2.2,topL+.2,0.05,0,'$MDD_1 < MDD_5$')
plt.ylabel("A' difference: (neutral - negative)")
plt.xlabel("Visit")
plt.savefig('thesis_plots/aprime_sadbias.pdf')
plt.show()

stat = all_happybias
fig,ax = plotPosterStyle_DF(stat,subjects)
plt.ylim([-.6,0.6])
plt.text(-.4,.6,'harder to attend\nto positive faces',fontsize=10,va='top',ha='left')
plt.text(-.4,-.6,'harder to attend\nto netural faces',fontsize=10,va='bottom', ha='left')
plt.xticks(np.arange(4),('Pre NF', 'Mid NF', 'Post NF', '1M FU'))
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,topL,0.05,0,'$MDD > HC$')
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,2.2,topL+.2,0.05,0,'$MDD_1 < MDD_5$')
plt.ylabel("A' difference: (neutral - positive)")
plt.xlabel("Visit")
plt.savefig('thesis_plots/aprime_positivebias.pdf')
plt.show()


########################################################################################################################
# (2) - Eye tracking Data
first_orientation,total_viewing_time,fixation_durations_first_time = getGazeData(subjects)
emo = ['DYS', 'THR', 'NEU', 'POS']
DYSPHORIC = 1
THREAT = 2
NEUTRAL = 3
POSITIVE = 4

stat = np.nanmean(first_orientation[:,:,:,DYSPHORIC-1],axis=1)
topL=0.3
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
plt.ylim([0,.65])
plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))
x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,0.4,0.05,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,topL,0.05,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,1.8,2.2,topL,0.05,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,2.2,topL+.2,0.05,0,'$MDD_1 > MDD_5$')

x,y = nonNan(stat[HC_ind,3],stat[MDD_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,2.8,3.2,topL,0.05,0,'$MDD > HC$')
plt.ylabel('ratio first orientation - dysphoric')
plt.xlabel('Visit')
plt.savefig('thesis_plots/gaze_orientation_dysphoric.pdf')
plt.show()

stat = np.nanmean(first_orientation[:,:,:,POSITIVE-1],axis=1)
topL=0.3
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
plt.ylim([0,.65])
plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))
x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,0.4,0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,0.8,1.2,topL,0.05,0,r'$MDD \neq HC$')

x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,1.8,2.2,topL,0.05,0,r'$MDD \neq HC$')
x,y = nonNan(stat[HC_ind,3],stat[MDD_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,2.8,3.2,topL,0.05,0,r'$MDD \neq HC$')
plt.ylabel('ratio first orientation - positive')
plt.xlabel('Visit')
plt.savefig('thesis_plots/gaze_orientation_positive.pdf')
plt.show()

stat = np.nanmean(total_viewing_time[:,:,:,DYSPHORIC-1],axis=1)
topL=0.3
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
plt.ylim([0,.6])
plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))
x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,0.4,0.025,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,topL,0.025,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,1.8,2.2,topL,0.025,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,2.2,topL+.25,0.025,0,'$MDD_1 > MDD_5$')

x,y = nonNan(stat[HC_ind,3],stat[MDD_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,2.8,3.2,topL,0.025,0,'$MDD > HC$')
plt.ylabel('ratio total time - dysphoric')
plt.xlabel('Visit')
plt.savefig('thesis_plots/gaze_totalviewing_dysphoric.pdf')
plt.show()

# relationship to MADRS?
data_mat = np.nanmean(total_viewing_time[:,:,:,DYSPHORIC-1],axis=1)
all_neg_change = data_mat[:,2] - data_mat[:,0]
#all_neg_change = specifically_neg
colors = ['k', 'r'] # HC, MDD
colors = ['#636363','#de2d26']
#fig = plt.figure(figsize=(10,7))
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
plt.xlabel('improvement in negative viewing time',fontsize=20)
plt.ylabel('improvement in depression severity',fontsize=20)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim([-.5,.8])
plt.title('MADRS vs. negative viewing time Change')
x,y = nonNan(-1*all_neg_change,-1*d1)
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,20,text, ha='left',va='top',color='k',fontsize=25)
x,y = nonNan(-1*all_neg_change[MDD_ind],-1*d1[MDD_ind])
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,15,text, ha='left',va='top',color='k',fontsize=25)
#labels_pos_v = np.array([-0.4,0,0.4,0.8])
#labels_pos = labels_pos_v.astype(np.str)
#plt.xticks(labels_pos_v,labels_pos,fontsize=30)
plt.savefig('thesis_plots/total_viewing_vs_MADRS.pdf')
plt.show()



# look for interaction
stat = np.nanmean(total_viewing_time[:,:,:,DYSPHORIC-1],axis=1)
df = convertMatToDF(stat,subjects)
df2 = df[(df['day']==0) | (df['day'] == 2)]
model = ols('data ~ group*day',df2).fit()
model.summary()

stat = np.nanmean(total_viewing_time[:,:,:,POSITIVE-1],axis=1)
topL=0.3
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
plt.ylim([0,.8])
plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))
x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,0.4,0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,0.8,1.2,topL,0.05,0,r'$MDD \neq HC$')

x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,1.8,2.2,topL,0.05,0,r'$MDD \neq HC$')

# x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
# t,p = scipy.stats.ttest_rel(x,y)
# addComparisonStat_SYM(p/2,0.2,2.2,topL+.2,0.05,0,'$MDD_1 < MDD_5$')

x,y = nonNan(stat[HC_ind,3],stat[MDD_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,2.8,3.2,topL,0.05,0,r'$MDD \neq HC$')
plt.ylabel('ratio total time - positive')
plt.xlabel('Visit')
plt.savefig('thesis_plots/gaze_totalviewing_positive.pdf')
plt.show()

stat = np.nanmean(fixation_durations_first_time[:,:,:,DYSPHORIC-1],axis=1)
topL=0.27
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
plt.ylim([0,.4])
plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))
x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,topL,0.025,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,topL,0.025,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,1.8,2.2,topL,0.025,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,2.2,topL+.1,0.025,0,'$MDD_1 > MDD_5$')

x,y = nonNan(stat[HC_ind,3],stat[MDD_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,2.8,3.2,topL,0.025,0,'$MDD > HC$')
plt.ylabel('dwell time first fixation - dysphoric')
plt.xlabel('Visit')
plt.savefig('thesis_plots/gaze_firstfix_dysphoric.pdf')
plt.show()

# look for interaction
stat = np.nanmean(fixation_durations_first_time[:,:,:,DYSPHORIC-1],axis=1)
df = convertMatToDF(stat,subjects)
df2 = df[(df['day']==0) | (df['day'] == 2)]
model = ols('data ~ group*day',df2).fit()
model.summary()



data_mat =  np.nanmean(fixation_durations_first_time[:,:,:,DYSPHORIC-1],axis=1)
all_neg_change = data_mat[:,2] - data_mat[:,0]
#all_neg_change = specifically_neg
colors = ['k', 'r'] # HC, MDD
colors = ['#636363','#de2d26']
#fig = plt.figure(figsize=(10,7))
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
plt.xlabel('improvement in negative maintenance',fontsize=20)
plt.ylabel('improvement in depression severity',fontsize=20)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim([-.1,.2])
plt.title('MADRS vs. dysphoric maintenance Change')
x,y = nonNan(-1*all_neg_change,-1*d1)
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.27,20,text, ha='left',va='top',color='k',fontsize=25)
x,y = nonNan(-1*all_neg_change[MDD_ind],-1*d1[MDD_ind])
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.27,15,text, ha='left',va='top',color='k',fontsize=25)
#labels_pos_v = np.array([-0.4,0,0.4,0.8])
#labels_pos = labels_pos_v.astype(np.str)
#plt.xticks(labels_pos_v,labels_pos,fontsize=30)
plt.savefig('thesis_plots/first_fix_vs_MADRS.pdf')
plt.show()


sns.barplot(data=df2,x='day',y='data',hue='group')

stat = np.nanmean(fixation_durations_first_time[:,:,:,POSITIVE-1],axis=1)
topL=0.3
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,1,2,3])],subjects)
plt.ylim([0,.5])
plt.xticks(np.arange(4),('Pre NF', 'Mid NF' ,'Post NF', '1M FU'))
x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,topL,0.05,0,'$MDD < HC$')

x,y = nonNan(stat[MDD_ind,1],stat[HC_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,0.8,1.2,topL,0.05,0,r'$MDD \neq HC$')

x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,1.8,2.2,topL,0.05,0,r'$MDD \neq HC$')

# x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
# t,p = scipy.stats.ttest_rel(x,y)
# addComparisonStat_SYM(p/2,0.2,2.2,topL+.2,0.05,0,'$MDD_1 < MDD_5$')

x,y = nonNan(stat[HC_ind,3],stat[MDD_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,2.8,3.2,topL,0.05,0,r'$MDD \neq HC$')
plt.ylabel('dwell time first fixation - positive')
plt.xlabel('Visit')
plt.savefig('thesis_plots/gaze_firstfix_positive.pdf')
plt.show()

########################################################################################################################

# QUESTION 1 - IS NETWORK CORRELATION RELATED TO TOAL A' DIFFERENCES?
perception_correlation, attention_correlation, perception_attention_correlation = correlateROICS(subjects,0)
perception_run_avg = np.nanmean(perception_correlation,axis=1)
perception_avg = np.nanmean(perception_run_avg,axis=1)
attention_run_avg = np.nanmean(attention_correlation,axis=1)
attention_avg = np.nanmean(attention_run_avg,axis=1)


# run behavioral A' results too! *********

### FIRST REPEAT MEGAN'S ANALYSIS

# so this is ALL NEUROFEEDBACK RUNS!! correlation with ALL behavior
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
plt.xlabel('correlation: wb - perceptual network',fontsize=20)
plt.ylabel(r'behavioral training $\Delta$A',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=20)
plt.xlim([0.45,0.9])
plt.ylim([-.2,.25])
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(.47,.21,text, ha='left',va='top',color='k',fontsize=20)
x1,y1 = nonNan(x[MDD_ind],y[MDD_ind])
r,p=scipy.stats.pearsonr(x1,y1)
text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(.47,.1,text, ha='left',va='top',color='k',fontsize=20)
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
plt.xlabel('correlation: wb - attentional network',fontsize=20)
plt.ylabel('')
plt.xticks(fontsize=15)
plt.xlim([0.45,0.9])
plt.ylim([-.2,.25])
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(.8,.21,text, ha='left',va='top',color='k',fontsize=20)#plt.ylim([-.2,.2])
x1,y1 = nonNan(x[MDD_ind],y[MDD_ind])
r,p=scipy.stats.pearsonr(x1,y1)
text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(0.8,.1,text, ha='left',va='top',color='k',fontsize=20)
#plt.title('Transitions %i shifted ahead' % nshift1)
plt.savefig('thesis_plots/correlation_networks_original_way.pdf')
plt.show()


#y = (all_happyBlocks[:,2] - all_happyBlocks[:,0])# the higher the sadbias, the worse you do when sad faces are distracting
#y = correct_stickiness
#y = combined_diff
# so if the difference decreases, that's good -- let's take negative

x1,y1 = nonNan(x[MDD_ind],y[MDD_ind])
r,p=scipy.stats.pearsonr(x1,y1)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
print(text)

x1,y1 = nonNan(x[HC_ind],y[HC_ind])
r,p=scipy.stats.pearsonr(x1,y1)
text='\nHC only\nr = %2.2f\np = %2.2f' % (r,p)
print(text)


x1,y1 = nonNan(x,y)
r,p=scipy.stats.pearsonr(x1,y1)
text='\nALL subjects\nr = %2.2f\np = %2.2f' % (r,p)
print(text)


#### QUESTION 2 - NF IMPROVEMENT - take half this time
perception_correlationH, attention_correlationH, perception_attention_correlationH = correlateROICS(subjects,1)
perception_run_avgH = np.nanmean(perception_correlationH,axis=1)
perception_change = perception_run_avgH[:,2] - perception_run_avgH[:,0]
attention_run_avgH = np.nanmean(attention_correlationH,axis=1)
attention_change = attention_run_avgH[:,2] - attention_run_avgH[:,0]
master_x,all_x = buildMasterDict(subjects,1)
bins = [-1.   , -0.975, -0.9, -0.8 ,-0.7,-0.55,-0.4,-0.2,0,0.2,0.4,0.55,0.7, 0.8 ,  0.9 , 0.975, 1. ]
all_matrices,p_state,subject_averages = buildTransitionMatrix(subjects,all_x,bins,2,3)
# get difference post - pre
neg_stickiness = all_matrices[0,0,:,2] - all_matrices[0,0,:,0]
correct_stickiness = all_matrices[15,15,:,2] - all_matrices[15,15,:,0]



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
plt.xlabel(r'$\Delta$correlation: wb - perceptual network',fontsize=20)
plt.ylabel(r'behavioral training $\Delta$A',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=20)
#plt.xlim([0.45,0.9])
plt.ylim([-.2,.25])
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,.21,text, ha='left',va='top',color='k',fontsize=20)
r,p=scipy.stats.pearsonr(x[MDD_ind],y[MDD_ind])
text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(0,.21,text, ha='left',va='top',color='k',fontsize=20)
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
plt.xlabel(r'$\Delta$correlation: wb - attentional network',fontsize=20)
plt.ylabel('')
plt.xticks(fontsize=15)
#plt.xlim([0.45,0.9])
plt.ylim([-.2,.25])
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,.21,text, ha='left',va='top',color='k',fontsize=20)#plt.ylim([-.2,.2])
r,p=scipy.stats.pearsonr(x[MDD_ind],y[MDD_ind])
text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(0.2,.21,text, ha='left',va='top',color='k',fontsize=20)
#plt.title('Transitions %i shifted ahead' % nshift1)
plt.savefig('thesis_plots/correlation_networks_change_corr.pdf')
plt.show()


fig,ax = plt.subplots(figsize=(20,10))
plt.subplot(1,2,1)
x = perception_change
y = sadBlocks_diff 
sns.despine()
for s in np.arange(nsubs):
  subjectNum  = subjects[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(x[s],y[s],marker='.',ms=30,color=colors[style],alpha=0.9)
plt.xlabel(r'$\Delta$correlation: wb - perceptual network',fontsize=20)
plt.ylabel(r'behavioral training $\Delta$A',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=20)
#plt.xlim([0.45,0.9])
plt.ylim([-.2,.25])
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,.21,text, ha='left',va='top',color='k',fontsize=20)
r,p=scipy.stats.pearsonr(x[MDD_ind],y[MDD_ind])
text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,.15,text, ha='left',va='top',color='k',fontsize=20)

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
plt.xlabel(r'$\Delta$correlation: wb - attentional network',fontsize=20)
plt.ylabel('')
plt.xticks(fontsize=15)
#plt.xlim([0.45,0.9])
plt.ylim([-.2,.25])
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,.21,text, ha='left',va='top',color='k',fontsize=20)#plt.ylim([-.2,.2])
r,p=scipy.stats.pearsonr(x[MDD_ind],y[MDD_ind])
text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,.15,text, ha='left',va='top',color='k',fontsize=20)
#plt.title('Transitions %i shifted ahead' % nshift1)
plt.savefig('thesis_plots/correlation_networks_change_corr_distract_sad.pdf')
plt.show()



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
plt.xlabel(r'$\Delta$correlation: wb - perceptual network',fontsize=20)
plt.ylabel(r'$\Delta$ stickiness (improvement)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=20)
#plt.xlim([0.45,0.9])
#plt.ylim([-.2,.25])
x1,y1 = nonNan(x,y)
r,p=scipy.stats.pearsonr(x1,y1)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,.6,text, ha='left',va='top',color='k',fontsize=20)
x1,y1 = nonNan(x[MDD_ind],y[MDD_ind])
r,p=scipy.stats.pearsonr(x1,y1)
text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,.3,text, ha='left',va='top',color='k',fontsize=20)
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
plt.xlabel(r'$\Delta$correlation: wb - attentional network',fontsize=20)
plt.ylabel('')
plt.xticks(fontsize=15)
#plt.xlim([0.45,0.9])
#plt.ylim([-.2,.25])
x1,y1 = nonNan(x,y)
r,p=scipy.stats.pearsonr(x1,y1)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(.2,.6,text, ha='left',va='top',color='k',fontsize=20)
x1,y1 = nonNan(x[MDD_ind],y[MDD_ind])
r,p=scipy.stats.pearsonr(x1,y1)
text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(.2,.3,text, ha='left',va='top',color='k',fontsize=20)
#plt.title('Transitions %i shifted ahead' % nshift1)
plt.savefig('thesis_plots/correlation_networks_change_corr_stickiness.pdf')
plt.show()



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
plt.xlabel(r'$\Delta$correlation: wb - perceptual network',fontsize=20)
plt.ylabel(r'$\Delta$ correct attention',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=20)
#plt.xlim([0.45,0.9])
#plt.ylim([-.2,.25])
x1,y1 = nonNan(x,y)
r,p=scipy.stats.pearsonr(x1,y1)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.18,.6,text, ha='left',va='top',color='k',fontsize=20)
x1,y1 = nonNan(x[MDD_ind],y[MDD_ind])
r,p=scipy.stats.pearsonr(x1,y1)
text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.18,.4,text, ha='left',va='top',color='k',fontsize=20)
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
plt.xlabel(r'$\Delta$correlation: wb - attentional network',fontsize=20)
plt.ylabel('')
plt.xticks(fontsize=15)
#plt.xlim([0.45,0.9])
#plt.ylim([-.2,.25])
x1,y1 = nonNan(x,y)
r,p=scipy.stats.pearsonr(x1,y1)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,.6,text, ha='left',va='top',color='k',fontsize=20)
x1,y1 = nonNan(x[MDD_ind],y[MDD_ind])
r,p=scipy.stats.pearsonr(x1,y1)
text='MDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,.4,text, ha='left',va='top',color='k',fontsize=20)
#plt.title('Transitions %i shifted ahead' % nshift1)
plt.savefig('thesis_plots/correlation_networks_change_corr_correct_attention.pdf')
plt.show()


x =attention_change
#y = (all_happyBlocks[:,2] - all_happyBlocks[:,0])# the higher the sadbias, the worse you do when sad faces are distracting
y = neg_stickiness
#y = combined_diff_neutral
# so if the difference decreases, that's good -- let's take negative
fig,ax = plt.subplots(figsize=(12,10))
sns.despine()
for s in np.arange(nsubs):
  subjectNum  = subjects[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(x[s],y[s],marker='.',ms=30,color=colors[style],alpha=0.9)
plt.xlabel('perception correlation change',fontsize=20)
plt.ylabel('improvement in MADRS',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.ylim([-.2,.2])
plt.xlim([-.2,.3])
#plt.title('Transitions %i shifted ahead' % nshift1)
plt.show()

x1,y1 = nonNan(x[MDD_ind],y[MDD_ind])
r,p=scipy.stats.pearsonr(x1,y1)
text='\nMDD only\nr = %2.2f p = %2.2f' % (r,p)
print(text)

x1,y1 = nonNan(x,y)
r,p=scipy.stats.pearsonr(x1,y1)
text='\nALL subjects\nr = %2.2f p = %2.2f' % (r,p)
print(text)

######################################################### TRANSITION MATRIX ANALYSIS ######################################
master_x,all_x = buildMasterDict(subjects,1)
bins = [-1.   , -0.975, -0.9, -0.8 ,-0.7,-0.55,-0.4,-0.2,0,0.2,0.4,0.55,0.7, 0.8 ,  0.9 , 0.975, 1. ]
nbins=len(bins)

### FIRST MAKE MEGA HISTOGRAM
fig,ax = plt.subplots(figsize=(18,11))
sns.despine()
#n = plt.hist(master_x,bins)
counts, bins = np.histogram(master_x,bins)
total_val = np.sum(counts)
counts_norm = counts/total_val
plt.hist(bins[:-1], bins, weights=counts_norm,color='k',alpha=0.25,edgecolor='k',linewidth=3)
#plt.bar(bins[:-1],counts_norm,color='k',alpha=0.25,edgecolor='k', linewidth=3)
labels_pos = np.array(bins).astype(np.float)
labels_pos = np.around(labels_pos,decimals=2).astype(np.str)
for l in np.arange(len(labels_pos)):
  # alternate every other label
  if np.mod(l,2):
    labels_pos[l] = '\n' + labels_pos[l]
plt.xticks(bins,labels_pos,fontsize=13)
plt.ylabel('ratio of total counts')
plt.xlim([-1,1])
plt.ylim([0,.1])
plt.xlabel('(scene-face) bin')
plt.savefig('thesis_plots/nf_histogram.pdf')
plt.show()

all_matrices,p_state,subject_averages = buildTransitionMatrix(subjects,all_x,bins,2,3)
nDays=3
diagonal_data = np.zeros((nsubs,nbins-1,nDays))
labels_pos = np.array(bins).astype(np.float)
labels_pos = np.around(labels_pos,decimals=2).astype(np.str)
for s in np.arange(nsubs):
    for d in np.arange(nDays):
        ex = all_matrices[:,:,s,d]
        diagonal = np.diagonal(ex)
        diagonal_data[s,:,d] = diagonal
# find largest group diff
diag_diff = np.abs(np.nanmean(diagonal_data[MDD_ind,:,0],axis=0) - np.nanmean(diagonal_data[HC_ind,:,0],axis=0))
np.argmax(diag_diff)
diag_diff = np.abs(np.nanmean(diagonal_data[MDD_ind,:,2],axis=0) - np.nanmean(diagonal_data[HC_ind,:,2],axis=0))
np.argmax(diag_diff)
# now plot for day 1 
fig = plotPosterStyle_multiplePTS(diagonal_data[:,:,np.array([0,2])],subjects,0)
plt.subplot(1,2,1)
plt.ylim([0,.55])
plt.yticks(np.linspace(0,.5,3),fontsize=25)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xlabel('bin at t')
plt.ylabel('p(stay in bin) at t+5 s')
plt.title('Early NF')
x,y = nonNan(diagonal_data[MDD_ind,0,0],diagonal_data[HC_ind,0,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0,0,0.5,0.01,0,'')
plt.subplot(1,2,2)
#plt.ylim([0,.4])
plt.ylim([0,.55])
plt.yticks(np.linspace(0,.5,3),fontsize=25)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
x,y = nonNan(diagonal_data[MDD_ind,15,2],diagonal_data[HC_ind,15,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0,0,0.5,0.01,0,'')
plt.legend()
plt.xlabel('bin at t')
plt.title('Late NF')
plt.savefig('thesis_plots/nf_diagonal.pdf')
plt.show()


# CHECK INTERACTIONS
i=0
j=0
stat = all_matrices[i,j,:,:]
df = convertMatToDF(stat,subjects)
model = ols('data ~ group*day',df).fit()
model.summary()

i=15
j=15
stat = all_matrices[i,j,:,:]
df = convertMatToDF(stat,subjects)
model = ols('data ~ group*day',df).fit()
model.summary()

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
# add interaction
addComparisonStat_SYM(0.045,0,1,np.nanmax(stat)+.45,0,0,'group:visit',0)
plt.ylabel('p(stay in saddest state)',fontsize=40)
plt.ylim([0,1.25])
plt.xticks(np.arange(2),('Early NF','Late NF'),fontsize=30)
labels_pos_v = np.array([0,0.5,1])
labels_pos = labels_pos_v.astype(np.str)
plt.yticks(labels_pos_v,labels_pos,fontsize=30)
plt.xlabel('')
plt.savefig('thesis_plots/nf_saddest_bargraph.pdf')
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
plt.ylim([0,1.25])
plt.xticks(np.arange(2),('Early NF','Late NF'),fontsize=30)
labels_pos_v = np.array([0,0.5,1])
labels_pos = labels_pos_v.astype(np.str)
plt.yticks(labels_pos_v,labels_pos,fontsize=30)
plt.xlabel('')
plt.savefig('thesis_plots/nf_correct_bargraph.pdf')
plt.show()

x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)

data_mat = all_matrices[0,0,:,:]
all_neg_change = data_mat[:,2] - data_mat[:,0]
#all_neg_change = specifically_neg
colors = ['k', 'r'] # HC, MDD
colors = ['#636363','#de2d26']
#fig = plt.figure(figsize=(10,7))
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
plt.xlabel('improvement in negative stickiness',fontsize=20)
plt.ylabel('improvement in depression severity',fontsize=20)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim([-.5,.8])
plt.title('MADRS vs. NF Change')
x,y = nonNan(-1*all_neg_change,-1*d1)
r,p=scipy.stats.pearsonr(x,y)

text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.47,20,text, ha='left',va='top',color='k',fontsize=25)
x,y = nonNan(-1*all_neg_change[MDD_ind],-1*d1[MDD_ind])
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.47,15,text, ha='left',va='top',color='k',fontsize=25)
labels_pos_v = np.array([-0.4,0,0.4,0.8])
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(labels_pos_v,labels_pos,fontsize=30)
plt.savefig('thesis_plots/NF_vs_MADRS.pdf')
plt.show()

# general averages by day
stat = subject_averages
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,2])],subjects)
x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD_1 < HC_1$')
x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0)
# x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
# t,p = scipy.stats.ttest_rel(x,y)
# addComparisonStat_SYM(p/2,0.2,1.2,np.nanmax(stat)+.2,0.05,0,'$MDD_1 > MDD_3$')
plt.ylabel('mean(p(scene) - p(face))',fontsize=30)
plt.ylim([-.5,.5])
plt.xticks(np.arange(2),('NF 1','NF 3'),fontsize=30)
labels_pos_v = np.array([-.5,0,0.5])
labels_pos = labels_pos_v.astype(np.str)
plt.yticks(labels_pos_v,labels_pos,fontsize=30)
plt.xticks(np.arange(2),('Early NF','Late NF'),fontsize=30)
plt.xlabel('')
#plt.savefig('thesis_plots/nf_clf_avg.pdf')
plt.show()

# do both groups decrease significantly?
x,y = nonNan(stat[:,0],stat[:,2])
t,p = scipy.stats.ttest_rel(x,y)

# TO DO: HISTOGRAM PLOT

######################################################### RESTING STATE ANALYSIS - NODE ######################################
func_con_subjects = np.array([3,4,5,6,7,8,9,10,11,12,106,107,108,109,110,111,112,113,114,115])
HC_ind_con = np.argwhere(func_con_subjects<100)[:,0]
MDD_ind_con = np.argwhere(func_con_subjects>100)[:,0]
average_within_mat, amyg_con = getFunctionalCon(func_con_subjects)

dmn_connectivity = average_within_mat[0,0,:,:] # subjects x visit
stat = dmn_connectivity
topL=0.2
fig,ax = plotPosterStyle_DF(stat,func_con_subjects)
plt.ylim([0,.5])
plt.xticks(np.arange(2),('Pre NF', 'Post NF'))
x,y = nonNan(stat[MDD_ind_con,0],stat[HC_ind_con,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,topL,0.05,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind_con,1],stat[HC_ind_con,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,topL,0.05,0,'$MDD > HC$')

x,y = nonNan(stat[MDD_ind_con,0],stat[MDD_ind_con,1])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,1.2,topL+.15,0.05,0,'$MDD_1 > MDD_5$')
plt.ylabel('within network connectivity - DMN')
plt.xlabel('Visit')
plt.savefig('thesis_plots/con_dmn.pdf')
plt.show()

data_mat =dmn_connectivity
all_neg_change1 = data_mat[:,1] - data_mat[:,0]
all_subjects_func_con = np.zeros((nsubs,)) * np.nan
for s in np.arange(nsubs):
  if subjects[s] in func_con_subjects:
    ind_f = np.argwhere(subjects[s] == func_con_subjects)[0][0]
    all_subjects_func_con[s] = all_neg_change1[ind_f]
all_neg_change = all_subjects_func_con
#all_neg_change = specifically_neg
colors = ['k', 'r'] # HC, MDD
colors = ['#636363','#de2d26']
#fig = plt.figure(figsize=(10,7))
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
plt.xlabel('improvement in negative stickiness',fontsize=20)
plt.ylabel('improvement in depression severity',fontsize=20)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim([-.5,.8])
plt.title('MADRS vs. NF Change')
x,y = nonNan(-1*all_neg_change,-1*d1)
r,p=scipy.stats.pearsonr(x,y)

text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.47,20,text, ha='left',va='top',color='k',fontsize=25)
x,y = nonNan(-1*all_neg_change[MDD_ind],-1*d1[MDD_ind])
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.47,15,text, ha='left',va='top',color='k',fontsize=25)
# labels_pos_v = np.array([-0.4,0,0.4,0.8])
# labels_pos = labels_pos_v.astype(np.str)
# plt.xticks(labels_pos_v,labels_pos,fontsize=30)
plt.savefig('thesis_plots/DMN_vs_MADRS.pdf')
plt.show()


fpn_connectivity = average_within_mat[1,1,:,:] # subjects x visit
stat = fpn_connectivity
topL=0.2
fig,ax = plotPosterStyle_DF(stat,func_con_subjects)
plt.ylim([0,.5])
plt.xticks(np.arange(2),('Pre NF', 'Post NF'))
x,y = nonNan(stat[MDD_ind_con,0],stat[HC_ind_con,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,-.2,.2,topL,0.05,0,r'$MDD \neq HC$')

x,y = nonNan(stat[MDD_ind_con,1],stat[HC_ind_con,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,0.8,1.2,topL,0.05,0,r'$MDD \neq HC$')

x,y = nonNan(stat[MDD_ind_con,0],stat[MDD_ind_con,1])
t,p = scipy.stats.ttest_rel(x,y)

# x,y = nonNan(stat[MDD_ind_con,0],stat[MDD_ind_con,1])
# t,p = scipy.stats.ttest_rel(x,y)
# addComparisonStat_SYM(p/2,0.2,1.2,topL+.3,0.05,0,'$MDD_1 > MDD_5$')
plt.ylabel('within network connectivity - FPN')
plt.xlabel('Visit')
plt.savefig('thesis_plots/con_fpn.pdf')
plt.show()

dmn_to_fpn = average_within_mat[1,0,:,:]
stat = dmn_to_fpn
topL=np.nanmax(stat)
fig,ax = plotPosterStyle_DF(stat,func_con_subjects)
#plt.ylim([0,.8])
plt.xticks(np.arange(2),('Pre NF', 'Post NF'))
x,y = nonNan(stat[MDD_ind_con,0],stat[HC_ind_con,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,-.2,.2,topL,0.00005,0,r'$MDD \neq HC$')

x,y = nonNan(stat[MDD_ind_con,1],stat[HC_ind_con,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,0.8,1.2,topL,0.00005,0,r'$MDD \neq HC$')

# x,y = nonNan(stat[MDD_ind_con,0],stat[MDD_ind_con,1])
# t,p = scipy.stats.ttest_rel(x,y)
# addComparisonStat_SYM(p/2,0.2,1.2,topL+.3,0.05,0,'$MDD_1 > MDD_5$')
plt.ylabel('DMN - FPN connectivity')
plt.xlabel('Visit')
ax.ticklabel_format(axis='y', style='scientific',scilimits=(-2,2))
plt.savefig('thesis_plots/con_dmn_fpn.pdf')
plt.show()


# check interaction
stat = dmn_to_fpn
df = convertMatToDF(stat,func_con_subjects)
model = ols('data ~ group*day',df).fit()
model.summary()
######################################################### RESTING STATE ANALYSIS - ROI ######################################
stat = amyg_con
topL=0.3
fig,ax = plotPosterStyle_DF(stat,func_con_subjects)
#plt.ylim([0,.8])
plt.xticks(np.arange(2),('Pre NF', 'Post NF'))
x,y = nonNan(stat[MDD_ind_con,0],stat[HC_ind_con,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,-.2,.2,0.4,0.05,0,r'$MDD \neq HC$')

x,y = nonNan(stat[MDD_ind_con,1],stat[HC_ind_con,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,0.8,1.2,topL,0.05,0,r'$MDD \neq HC$')

# x,y = nonNan(stat[MDD_ind_con,0],stat[MDD_ind_con,1])
# t,p = scipy.stats.ttest_rel(x,y)
# addComparisonStat_SYM(p/2,0.2,1.2,topL+.3,0.05,0,'$MDD_1 > MDD_5$')
plt.ylabel('FPN - LA connectivity')
plt.xlabel('Visit')
plt.savefig('thesis_plots/con_fpn_amyg.pdf')
plt.show()
######################################################### FACES TASK ANALYSIS ############################################
colors_dark = ['#636363','#de2d26']
colors_light = ['#636363','#de2d26']
negative_ts, neutral_ts, happy_ts, nTR = getFaces3dTProjectData(subjects,'amyg_overlapping')
fig = plt.subplots(figsize=(19,10))
sns.despine()
x = np.arange(nTR)
day=0
y = negative_ts[HC_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[0], label='HC negative')
y = neutral_ts[HC_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC neutral')
y = negative_ts[MDD_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[1], label='MDD negative')
y = neutral_ts[MDD_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD neutral')
labels_pos_v = np.concatenate([np.arange(-5,nTR)])
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(np.arange(nTR),labels_pos,fontsize=30)
plt.text(5, 0.5, 'stim on', ha='center', va='bottom', color='k',fontsize=25)
plt.text(13, 0.5, 'stim off', ha='center', va='bottom', color='k',fontsize=25)
plt.ylabel('z-scored activity')
plt.xlabel('TR relative to block start')
plt.plot([5,5],[-10,10],'--', lw=1, c='k')
plt.plot([13,13],[-10,10],'--', lw=1, c='k')
plt.ylim([-.7,.7])
plt.legend()
plt.savefig('thesis_plots/faces_LA_day_1.pdf')
plt.show()

day=1
fig = plt.subplots(figsize=(19,10))
sns.despine()
y = negative_ts[HC_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[0], label='HC negative')
y = neutral_ts[HC_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC neutral')
y = negative_ts[MDD_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[1], label='MDD negative')
y = neutral_ts[MDD_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD neutral')
labels_pos_v = np.concatenate([np.arange(-5,nTR)])
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(np.arange(nTR),labels_pos,fontsize=30)
plt.text(5, 0.5, 'stim on', ha='center', va='bottom', color='k',fontsize=25)
plt.text(13, 0.5, 'stim off', ha='center', va='bottom', color='k',fontsize=25)
plt.ylabel('z-scored activity')
plt.xlabel('TR relative to block start')
plt.plot([5,5],[-10,10],'--', lw=1, c='k')
plt.plot([13,13],[-10,10],'--', lw=1, c='k')
plt.ylim([-.7,.7])
plt.legend()
plt.savefig('thesis_plots/faces_LA_day_3.pdf')
plt.show()


negative_diff = negative_ts - neutral_ts
# test interaction
stat = negative_diff[:,14,:]
df = convertMatToDF(stat,subjects)
model = ols('data ~ group*day',df).fit()
model.summary()

# for time only during the block - UNSHIFTED
# SKIP 5 + 2 = start at TR 7 --> 7 + 9 = 16
negative_diff_block = negative_diff[:,7:16,:]
day = 0
# average time series over block, now shifted
mdd_average = np.mean(negative_diff_block[MDD_ind,:,day],axis=1)
hc_average = np.mean(negative_diff_block[HC_ind,:,day],axis=1)
r,p = scipy.stats.ttest_ind(mdd_average,hc_average)

# now look at if you can let time vary there's an interaction - build df
negative_diff_initial = negative_diff_block[:,:,0]
negative_diff_final = negative_diff_block[:,:,1]

data = negative_diff_initial.flatten()
data2 = negative_diff_final.flatten()
both_data = np.concatenate((data,data2),axis=0)
nTR = 9
# goes through all subjects first
subjects = np.repeat(np.arange(nsubs),nTR)
subjects2 = np.concatenate((subjects,subjects),axis=0)

TR = np.tile(np.arange(nTR),nsubs)
TR2 = np.concatenate((TR,TR),axis=0)

group = [''] * len(subjects)
for i in np.arange(len(subjects)):
  if subjects[i] in MDD_ind:
    group[i] = 'MDD'
  elif subjects[i] in HC_ind:
    group[i] = 'HC'
group2 = group + group
day = np.repeat(np.arange(2),len(subjects))
all_data = {}
all_data['activity'] = data
all_data['TR'] = TR
all_data['group'] = group
all_data['subjects'] = subjects
df = pd.DataFrame.from_dict(all_data)
model = ols('activity ~ group*TR',df).fit()
model.summary()
all_data2 = {}
all_data2['activity'] = both_data
all_data2['TR'] = TR2
all_data2['group'] = group2
all_data2['subjects'] = subjects2
all_data2['day'] = day
df2 = pd.DataFrame.from_dict(all_data2)
model = ols('activity ~ group*day',df2).fit()
model.summary()
# difference between group
df3 = df2[df2['TR']==7]
model = ols('activity ~ group*day',df3).fit()
model.summary()


fig = plt.subplots(figsize=(19,10))
sns.despine()
x = np.arange(nTR)
day=0
y = negative_diff[HC_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[0], label='HC negative - neutral')

y = negative_diff[MDD_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[1], label='MDD negative - neutral')
labels_pos_v = np.concatenate([np.arange(-5,nTR)])
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(np.arange(nTR),labels_pos,fontsize=30)
plt.text(5, -.4, 'stim on', ha='center', va='bottom', color='k',fontsize=25)
plt.text(13, -.4, 'stim off', ha='center', va='bottom', color='k',fontsize=25)
plt.ylabel('z-scored activity difference')
plt.xlabel('TR relative to block start')
plt.plot([5,5],[-10,10],'--', lw=1, c='k')
plt.plot([13,13],[-10,10],'--', lw=1, c='k')
plt.ylim([-.7,.7])
plt.legend()
x,y = nonNan(negative_diff[MDD_ind,14,day],negative_diff[HC_ind,14,day])
t,p = scipy.stats.ttest_ind(x,y)
plt.legend(loc=2)
addComparisonStat_SYM(p/2,14,14,0.4,0.05,0,'$MDD > HC$')
plt.savefig('thesis_plots/faces_LA_diff_day_1.pdf')
plt.show()

x,y = nonNan(negative_diff[MDD_ind,14,1],negative_diff[HC_ind,14,1])
t,p = scipy.stats.ttest_ind(x,y)

x,y = nonNan(negative_diff[MDD_ind,14,0],negative_diff[MDD_ind,14,1])
t,p = scipy.stats.ttest_rel(x,y)

x,y = nonNan(negative_diff[HC_ind,14,0],negative_diff[HC_ind,14,1])
t,p = scipy.stats.ttest_rel(x,y)

fig = plt.subplots(figsize=(19,10))
sns.despine()
x = np.arange(nTR)
day=1
y = negative_diff[HC_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[0], label='HC negative - neutral')
y = negative_diff[MDD_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[1], label='MDD negative - neutral')
labels_pos_v = np.concatenate([np.arange(-5,nTR)])
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(np.arange(nTR),labels_pos,fontsize=30)
plt.text(5, -.4, 'stim on', ha='center', va='bottom', color='k',fontsize=25)
plt.text(13, -.4, 'stim off', ha='center', va='bottom', color='k',fontsize=25)
plt.ylabel('z-scored activity difference')
plt.xlabel('TR relative to block start')
plt.plot([5,5],[-10,10],'--', lw=1, c='k')
plt.plot([13,13],[-10,10],'--', lw=1, c='k')
plt.ylim([-.7,.7])
plt.legend()
x,y = nonNan(negative_diff[MDD_ind,14,day],negative_diff[HC_ind,14,day])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,14,14,0.4,0.05,0,'$MDD > HC$')
plt.legend(loc=2)
plt.savefig('thesis_plots/faces_LA_diff_day_3.pdf')
plt.show()


data_mat =  negative_diff[:,14,:]
all_neg_change = data_mat[:,1] - data_mat[:,0]
#all_neg_change = specifically_neg
colors = ['k', 'r'] # HC, MDD
colors = ['#636363','#de2d26']
#fig = plt.figure(figsize=(10,7))
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
x,y = nonNan(-1*all_neg_change[MDD_ind],-1*d1[MDD_ind])
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.27,15,text, ha='left',va='top',color='k',fontsize=25)
#labels_pos_v = np.array([-0.4,0,0.4,0.8])
#labels_pos = labels_pos_v.astype(np.str)
#plt.xticks(labels_pos_v,labels_pos,fontsize=30)
plt.savefig('thesis_plots/LA_change_vs_MADRS.pdf')
plt.show()

######################################################### FACES TASK ANALYSIS ############################################
