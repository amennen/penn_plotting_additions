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



subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101, 102,103,104,105,106, 107,108,109,110,111,112,113,114])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
nsubs = len(subjects)

# TO DO : TAKE INITIAL, MIDDLE, AND END AUC!! use that for controls 


rtAttenPath = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'
# get HC averages for each RUN OF SCANNER/DAY
nDays = 3
d1_runs = 6
d2_runs = 8
d3_runs = 7

takeHalf=0
perception_correlation = np.zeros((nsubs,d2_runs,nDays))*np.nan
attention_correlation = np.zeros((nsubs,d2_runs,nDays))*np.nan
perception_attention_correlation = np.zeros((nsubs,d2_runs,nDays))*np.nan

for s in np.arange(nsubs):
    subject_key = 'subject' + str(subjects[s])
    subjectDir = rtAttenPath + '/' + 'subject' + str(subjects[s])
    outfile = subjectDir + '/' + 'offlineAUC_RTCS.npz'  
    perception_outfile = subjectDir + '/' + 'offlineAUC_RTCS_perception.npz'   
    attention_outfile = subjectDir + '/' + 'offlineAUC_RTCS_attention.npz'  
    z=np.load(outfile)
    zp = np.load(perception_outfile)
    za = np.load(attention_outfile)
    if subjects[s] == 106:
        d1_runs = 5
    else:
        d1_runs = 6
    CS = z['csOverTime']
    CS_p = zp['csOverTime']
    CS_a = za['csOverTime']
    nTR = np.shape(CS)[1]
    for d in np.arange(nDays):
      if d == 0:
        if not takeHalf:
          d1_take = np.arange(d1_runs)
          nruns = d1_runs
        else:
          d1_take = np.arange(3)
          nruns = 3
      elif d == 1:
        if not takeHalf:
          d2_take = np.arange(d2_runs)
          nruns = d2_runs
        else:
          d2_take = np.array([2,3,4])
          nruns=3
      elif d == 2:
        if not takeHalf:
          d3_take = np.arange(d3_runs)
          nruns = d3_runs
        else:
          d3_take = np.arange(4,d3_runs)
          nruns = 3
      
      categSep = CS[0:nruns,:,d]
      categSep_p = CS_p[0:nruns,:,d]
      categSep_a = CS_a[0:nruns,:,d]
      for r in np.arange(nruns):
        original = categSep[r,:]
        perception = categSep_p[r,:]
        attention = categSep_a[r,:]
        perception_correlation[s,r,d] = scipy.stats.pearsonr(original,perception)[0]
        attention_correlation[s,r,d] = scipy.stats.pearsonr(original,attention)[0]
        perception_attention_correlation[s,r,d] = scipy.stats.pearsonr(attention,perception)[0]


data_mat = np.nanmean(attention_correlation,axis=1)
fig,ax = plotPosterStyle_DF(data_mat,subjects)
plt.ylim([0,1])
plt.xticks(np.arange(3),('NF 1', 'NF 2', 'NF 3'))
# x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
# t,p = scipy.stats.ttest_ind(x,y)
# addSingleStat(p/2,0,np.nanmax(stat),0.01)
# x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,2])
# t,p = scipy.stats.ttest_ind(x,y)
# addSingleStat(p/2,2,np.nanmax(stat),0.01)
# x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
# t,p = scipy.stats.ttest_rel(x,y)
# addComparisonStat(p/2,0,2,np.nanmax(stat),0.05)
plt.show()

# compare: overall correlation over all 3 days? to improvements in other metrics
overall_attention = np.nanmean(np.nanmean(attention_correlation,axis=1),axis=1)
attention_run_avg = np.nanmean(attention_correlation,axis=1)
attention_change = attention_run_avg[:,2] - attention_run_avg[:,0]
overall_perception = np.nanmean(np.nanmean(perception_correlation,axis=1),axis=1)
perception_run_avg = np.nanmean(perception_correlation,axis=1)
perception_change = perception_run_avg[:,2] - perception_run_avg[:,0]
stickiness_change = all_matrices[0,0,:,2] - all_matrices[0,0,:,0]
correct_change = all_matrices[15,15,:,2] - all_matrices[15,15,:,0]
pa_run = np.nanmean(perception_attention_correlation,axis=1)
pa_change = pa_run[:,2] - pa_run[:,0]
M = getMADRSscoresALL()
d1,d2,d3 = getMADRSdiff(M,subjects)
#all_neg_change = specifically_neg
colors = ['k', 'r'] # HC, MDD
colors = ['#636363','#de2d26']
fig,ax = plt.subplots(figsize=(12,10))
sns.despine()
for s in np.arange(nsubs):
#for s in keep:
  subjectNum  = subjects[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(overall_perception[s],-1*d1[s],marker='.',ms=30,color=colors[style])
plt.xlabel('change in perception correlation (post-pre)',fontsize=40)
plt.ylabel('improvement in depression severity',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim([-.5,.8])
plt.title('Transitions %i shifted ahead' % nshift1)
plt.show()


x,y = nonNan(overall_perception[MDD_ind],-1*stickiness_change[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.47,15,text, ha='left',va='top',color='k',fontsize=25)
labels_pos_v = np.array([-0.4,0,0.4,0.8])
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(labels_pos_v,labels_pos,fontsize=30)
#plt.savefig('poster_plots/MADRS_v_NF.png')
plt.show()
