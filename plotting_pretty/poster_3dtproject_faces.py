# PURPOSE: calculate raw percent signal change given specific ROI

import os
import glob
from shutil import copyfile
import json
import numpy as np
from subprocess import call
import sys
import scipy.stats
import nibabel as nib
import nilearn
from nilearn import image, masking
import pandas as pd
import matplotlib
import csv
import matplotlib.pyplot as plt
font = {'weight' : 'normal',
        'size'   : 22}
import pandas as pd
from anne_additions.plotting_pretty.commonPlotting import *
import seaborn as sns
matplotlib.rc('font', **font)
def convertTR(timing):    
    TR = np.floor(timing/2)
    TRint = int(TR)
    return TRint

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

def getRunResponse(category,run,all_start_timing,nTR,ROI_act):
  block1Start = int(all_start_timing[category,0,run])-5
  if block1Start < 0 * r:
    # pad with nans
    block1Start = 0
    block1 = np.concatenate((np.zeros(5,)*np.nan,ROI_act[block1Start:block1Start+nTR-5]))
  else:
    block1 = ROI_act[block1Start:block1Start+nTR]
  block2Start = int(all_start_timing[category,1,run])-5
  block3Start = int(all_start_timing[category,2,run])-5
  if block3Start+nTR > 142*(r+1): # if this spills over to the end of the run, don't include
      block3 = np.concatenate((ROI_act[block3Start:block3Start+nTR-5],np.zeros(5,)*np.nan))
  else:
      block3 = ROI_act[block3Start:block3Start+nTR]
  block2 = ROI_act[block2Start:block2Start+nTR]
  # take average over the whole response
  run_response = np.nanmean(np.concatenate((block1[:,np.newaxis],block2[:,np.newaxis],block3[:,np.newaxis]),axis=1),axis=1)
  return run_response


fmriprep_out="/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/fmriprep"
task_path = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/faces'
#save_path = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/normalized_runs_baseline'
save_path = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/highpass_normalized_runs_baseline'
amygdala_mask = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/mni_anat/LAMYG_in_MNI_overlapping.nii.gz'
timing_path = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/timing_files';
analyses_out = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/stats'
ROI_DIR = '/data/jux/cnds/amennen/rtAttenPenn/MNI_things/clusters'
# cluster indices 1& 2 is anterior cingulate  (dorsal & rostral)
# cluster index 9 is for amygdalaj
cluster=0
ROI = "{0}/cluster{1}sphere.nii.gz".format(ROI_DIR,cluster+1)
cluster = 9
amyg_cluster = "{0}/cluster{1}sphere.nii.gz".format(ROI_DIR,cluster+1)
all_categories = ['fearful','happy', 'neutral', 'object']
dorsal_acc = "{0}/cluster{1}sphere.nii.gz".format(ROI_DIR,0+1)
mask   = amygdala_mask
# BEFORE YOU DO ANYTHING SMOOTH THE DATA!
#subjectNum = np.int(sys.argv[1])
allsubjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115])
HC_ind = np.argwhere(allsubjects<100)[:,0]
MDD_ind = np.argwhere(allsubjects>100)[:,0]
sessions = [1,3]
nRuns = 2
trial=0
ntrials=6
# first just get negative
nsub = len(allsubjects)
# do ts of start + 24 s (12 TRs)
nTR = 9+10 # theres's 18 seconds = 9 TRs so 9 + 10 (5 before, 5 after)
ndays = len(sessions)
negative_ts = np.zeros((nsub,nTR,ndays))
neutral_ts = np.zeros((nsub,nTR,ndays))
happy_ts = np.zeros((nsub,nTR,ndays))

for s in np.arange(nsub):
    subjectNum = allsubjects[s]
    d=0
    bids_id = 'sub-{0:03d}'.format(subjectNum)
    print(bids_id)
    for ses in sessions:
        subjectDay = ses
        ses_id = 'ses-{0:02d}'.format(subjectDay)
        print(ses_id)
        day_path=os.path.join(analyses_out,bids_id,ses_id)

            # now load in timing (and convert to TR #)
        all_start_timing = np.zeros((len(all_categories),3,nRuns))
        for c in np.arange(len(all_categories)):
            category = all_categories[c]
            category_str = category + '.txt'
            file_name = os.path.join(timing_path,bids_id,ses_id, category_str)
            t = pd.read_fwf(file_name,header=None)
            timing = t.values # now 2 x 18 array
            all_start_timing[c,:,0] = np.array([convertTR(timing[0,trial]),convertTR(timing[0,trial+ntrials]),convertTR(timing[0,trial+(ntrials*2)])])
            all_start_timing[c,:,1] = np.array([convertTR(timing[1,trial]),convertTR(timing[1,trial+ntrials]),convertTR(timing[1,trial+(ntrials*2)])])+142
        run_response_neg = np.zeros((nRuns,nTR)) * np.nan
        run_response_neutral = np.zeros((nRuns,nTR)) * np.nan
        run_response_happy = np.zeros((nRuns,nTR)) * np.nan
        fn = glob.glob(os.path.join(day_path,'*_task-faces_glm_3dtproject.nii.gz'))
        output_img = fn[0]
        masked_img = nilearn.masking.apply_mask(output_img,mask)
        ROI_act = np.nanmean(masked_img,axis=1)

        # do for run 1 first
        for r in np.arange(nRuns):
          run_response_neg[r,:] = getRunResponse(0,r,all_start_timing,nTR,ROI_act)
          run_response_neutral[r,:] = getRunResponse(2,r,all_start_timing,nTR,ROI_act)
          run_response_happy[r,:] = getRunResponse(1,r,all_start_timing,nTR,ROI_act)

        negative_ts[s,:,d] = np.nanmean(run_response_neg,axis=0)
        neutral_ts[s,:,d] = np.nanmean(run_response_neutral,axis=0)
        happy_ts[s,:,d] = np.nanmean(run_response_happy,axis=0)
        d+=1

# TO DO: do block before and block afterwards
# NEXT TO DO: see if correlated to madrs
colors_dark = ['#636363','#de2d26']
colors_light = ['#636363','#de2d26']
fig = plt.subplots()
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
plt.show()

####### NOW SAME PLOT FOR HAPPY ###################################
colors_dark = ['#636363','#de2d26']
colors_light = ['#636363','#de2d26']
fig = plt.subplots()
sns.despine()
x = np.arange(nTR)
day=0
y = happy_ts[HC_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[0], label='HC happy')

y = neutral_ts[HC_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC neutral')


y = happy_ts[MDD_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[1], label='MDD happy')


y = neutral_ts[MDD_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD neutral')
labels_pos_v = np.concatenate([np.arange(-5,nTR)])
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(np.arange(nTR),labels_pos,fontsize=30)

plt.xlabel('TR relative to block start')
plt.plot([5,5],[-10,10],'--', lw=5, c='k')
plt.plot([13,13],[-10,10],'--', lw=5, c='k')

plt.ylim([-.7,.7])
plt.legend()
plt.show()



####### NOW SAME PLOT FOR DIFFERENCE ###################################
negative_diff = negative_ts - neutral_ts
fig = plt.subplots()
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
addComparisonStat_SYM(p/2,14,14,0.4,0.05,0,'$MDD > HC$')
plt.show()

####### NOW SAME PLOT FOR DIFFERENCE ###################################



response_diff_neg = negative_ts[:,14,:] - negative_ts[:,7,:]
change_time = response_diff_neg[:,1] - response_diff_neg[:,0]
#change_time = negative_ts[:,16,1] - negative_ts[:,16,0]
M = getMADRSscoresALL()
d1,d2,d3 = getMADRSdiff(M,allsubjects)
# negative bias change = RT neg bias 3 - RT neg bias 1
# for each day, mmore positive = more bias in negative direction, slower at negative lure (not really attending)
# so by multiplying by -1, we say how much they became less biased
colors = ['k', 'r'] # HC, MDD
colors = ['#636363','#de2d26']
#fig = plt.figure(figsize=(10,7))
# when you look at neurofeedback
#nf_change = data_mat[:,2] - data_mat[0]
fig,ax = plt.subplots(figsize=(12,10))
sns.despine()
for s in np.arange(nsub):
  subjectNum  = allsubjects[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(change_time[s],-1*d1[s],marker='.',ms=30,color=colors[style])
#plt.xlabel('improvement in negative stickiness',fontsize=40)
plt.xlabel('improvement in negative RT bias',fontsize=20)
plt.ylabel('improvement in MADRS',fontsize=20)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim([-.5,.8])
plt.title('')
# x,y = nonNan(-1*pos_bias_change,-1*negative_bias_change)
# r,p=scipy.stats.pearsonr(x,y)
# text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
# plt.text(-.12,15,text, ha='left',va='top',color='k',fontsize=25)
x,y = nonNan(-1*change_time[MDD_ind],-1*d1[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)
# text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
# plt.text(-.12,5,text, ha='left',va='top',color='k',fontsize=25)
# # labels_pos_v = np.array([-0.4,0,0.4,0.8])
# # labels_pos = labels_pos_v.astype(np.str)
# plt.xticks(labels_pos_v,labels_pos,fontsize=30)
#plt.savefig('poster_plots/MADRS_v_NF.png')
plt.show()




###################################
diff = negative_ts - neutral_ts
x = np.arange(nTR)
day=0
y = diff[HC_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[0], label='HC negative')

y = diff[MDD_ind,:,day]
ym = np.mean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0)
plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[1], label='MDD negative')

plt.show()

neutral_diff = np.nanmean(neutral_ts[MDD_ind,:,day],axis=0) - np.nanmean(neutral_ts[HC_ind,:,day],axis=0)
negative_diff = np.nanmean(negative_ts[MDD_ind,:,day],axis=0) - np.nanmean(negative_ts[HC_ind,:,day],axis=0)

x = np.arange(nTR)
day=0
y = negative_diff
ym = np.mean(y,axis=0)
plt.plot(y,color=colors_dark[1])
#yerr = scipy.stats.sem(y,axis=0)
#plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[1], label='HC negative')

y = neutral_diff
plt.plot(y,color=colors_dark[0])
#ym = np.mean(y,axis=0)
#yerr = scipy.stats.sem(y,axis=0)
#plt.errorbar(x,ym,yerr=yerr,linestyle='--',color=colors_dark[0], label='MDD negative')

plt.show()
