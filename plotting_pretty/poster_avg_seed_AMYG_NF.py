import os
import glob
from shutil import copyfile
import pandas as pd
import json
import numpy as np
from subprocess import call
import sys
import nilearn
from nilearn.image import new_img_like, load_img
from nilearn.input_data import NiftiMasker
from nilearn import plotting
import matplotlib.pyplot as plt
from anne_additions.plotting_pretty.commonPlotting import *


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

whole_brain_mask = '/data/jux/cnds/amennen/rtAttenPenn/MNI_things/mni_icbm152_t1_tal_nlin_asym_09c_BOLD_mask_Penn.nii'
amygdala_mask = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/mni_anat/LAMYG_in_MNI_overlapping.nii.gz'
corr_save_dir = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/neurofeedback/clean'


subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101, 102,103,104,105,106, 107,108,109,110,111,112,113,114])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
nSub = len(subjects)
nDays = 3
all_amygdala_correlations = np.zeros((9,nSub,nDays)) * np.nan
all_amygdala_correlations_scene = np.zeros((9,nSub,nDays)) * np.nan
all_amygdala_correlations_face = np.zeros((9,nSub,nDays)) * np.nan

# do: go through each subject,run,day and average LA mask --> make 1 matrix
for s in np.arange(nSub):
    subjectNum = subjects[s]
    bids_id = 'sub-{0:03d}'.format(subjectNum)
    print(bids_id)
    for d in np.arange(nDays):
        subjectDay=d+1
        ses_id = 'ses-{0:02d}'.format(subjectDay)
        print(ses_id)
        clean_path = corr_save_dir + '/' + bids_id + '/' + ses_id
        nRuns = len(glob.glob(clean_path+'/*CATEGSEP_seed_correlation_run*'))
        for r in np.arange(nRuns):
            run_id = 'run-{0:02d}'.format(r+2)
            print(run_id)
            full_filename_out = os.path.join(clean_path,'CATEGSEP_seed_correlation_'+run_id+'.nii.gz')
            amyg_corr = np.mean(nilearn.masking.apply_mask(full_filename_out,amygdala_mask))
            all_amygdala_correlations[r,s,d] = amyg_corr

            full_filename_out = os.path.join(clean_path,'CATEGSEP_seed_correlation_POS_'+run_id+'.nii.gz')
            amyg_corr = np.mean(nilearn.masking.apply_mask(full_filename_out,amygdala_mask))
            all_amygdala_correlations_scene[r,s,d] = amyg_corr

            full_filename_out = os.path.join(clean_path,'CATEGSEP_seed_correlation_NEG_'+run_id+'.nii.gz')
            amyg_corr = np.mean(nilearn.masking.apply_mask(full_filename_out,amygdala_mask))
            all_amygdala_correlations_face[r,s,d] = amyg_corr

# now take beginning of day 1, middle of day 2, end of day 3
day_averages = np.zeros((nSub,nDays))
day_averages[:,0] = np.nanmean(all_amygdala_correlations[np.arange(3),:,0],axis=0)
day_averages[:,1] = np.nanmean(all_amygdala_correlations[np.array([2,3,4]),:,1],axis=0)
day_averages[:,2] = np.nanmean(all_amygdala_correlations[np.arange(4,7),:,2],axis=0)

# all runs, all categsep
d1_runs = 6
d2_runs = 8
d3_runs = 7
day_averages = np.zeros((nSub,nDays))
day_averages[:,0] = np.nanmean(all_amygdala_correlations[np.arange(d1_runs),:,0],axis=0)
day_averages[:,1] = np.nanmean(all_amygdala_correlations[np.arange(d2_runs),:,1],axis=0)
day_averages[:,2] = np.nanmean(all_amygdala_correlations[np.arange(d3_runs),:,2],axis=0)



day_averages_pos = np.zeros((nSub,nDays))
day_averages_pos[:,0] = np.nanmean(all_amygdala_correlations_scene[np.arange(3),:,0],axis=0)
day_averages_pos[:,1] = np.nanmean(all_amygdala_correlations_scene[np.array([2,3,4]),:,1],axis=0)
day_averages_pos[:,2] = np.nanmean(all_amygdala_correlations_scene[np.arange(4,7),:,2],axis=0)

day_averages_neg = np.zeros((nSub,nDays))
day_averages_neg[:,0] = np.nanmean(all_amygdala_correlations_face[np.arange(3),:,0],axis=0)
day_averages_neg[:,1] = np.nanmean(all_amygdala_correlations_face[np.array([2,3,4]),:,1],axis=0)
day_averages_neg[:,2] = np.nanmean(all_amygdala_correlations_face[np.arange(4,7),:,2],axis=0)

day_averages_neg = np.zeros((nSub,nDays))
day_averages_neg[:,0] = np.nanmean(all_amygdala_correlations_face[np.arange(d1_runs),:,0],axis=0)
day_averages_neg[:,1] = np.nanmean(all_amygdala_correlations_face[np.arange(d2_runs),:,1],axis=0)
day_averages_neg[:,2] = np.nanmean(all_amygdala_correlations_face[np.arange(d3_runs),:,2],axis=0)


stat = day_averages_neg

stat=day_averages # both scene and face TRs
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
plt.xticks(np.arange(2),('Early NF','Late NF',))
#plt.xticks(np.arange(2),('All NF 1 Runs','All NF 3 Runs',))
plt.ylabel('LA correlation to S-F Evidence')
#plt.ylim([-.1,.4])
plt.show()



M = getMADRSscoresALL()
d1,d2,d3 = getMADRSdiff(M,subjects)
data_mat = day_averages
#all_neg_change = data_mat[:,0]
all_neg_change = data_mat[:,2] - data_mat[:,0]
colors = ['k', 'r'] # HC, MDD
colors = ['#636363','#de2d26']
#fig = plt.figure(figsize=(10,7))
fig,ax = plt.subplots(figsize=(12,10))
sns.despine()
for s in np.arange(nSub):
  subjectNum  = subjects[s]
  madrs_change = d1[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(-1*all_neg_change[s],-1*d1[s],marker='.',ms=30,color=colors[style])
plt.xlabel('increase in LA-NF NEGATIVE correlation (post-pre)',fontsize=15)
plt.ylabel('improvement in MADRS',fontsize=15)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
x,y = nonNan(-1*all_neg_change,-1*d1)
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.14,10,text, ha='left',va='top',color='k',fontsize=20)
x,y = nonNan(-1*all_neg_change[MDD_ind],-1*d1[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.14,7,text, ha='left',va='top',color='k',fontsize=20)


plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()



#plt.xlim([-.5,.8])
x,y = nonNan(all_neg_change,-1*d1)
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.47,20,text, ha='left',va='top',color='k',fontsize=25)
#MDD_ind = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  22, 23, 24])
x,y = nonNan(all_neg_change[MDD_ind],-1*d1[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.47,15,text, ha='left',va='top',color='k',fontsize=25)
labels_pos_v = np.array([-0.4,0,0.4,0.8])
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(labels_pos_v,labels_pos,fontsize=30)
#plt.savefig('poster_plots/MADRS_v_NF.png')

plt.show()


nf_change = all_matrices[0,0,:,:]
nf_by_day = nf_change[:,2] - nf_change[:,0]
fig,ax = plt.subplots(figsize=(12,10))
sns.despine()
for s in np.arange(nSub):
  subjectNum  = subjects[s]
  madrs_change = d1[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(all_neg_change[s],-1*nf_by_day[s],marker='.',ms=30,color=colors[style])
plt.xlabel('change in LA-NF correlation (post-pre)',fontsize=40)
plt.ylabel('improvement in negative stickiness',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()
x,y = nonNan(all_neg_change[MDD_ind],-1*nf_by_day[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)

amyg_change = diff_stat[:,1] - diff_stat[:,0]
fig,ax = plt.subplots(figsize=(12,10))
sns.despine()
for s in np.arange(nSub):
  subjectNum  = subjects[s]
  madrs_change = d1[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(all_neg_change[s],-1*amyg_change[s],marker='.',ms=30,color=colors[style])
plt.xlabel('change in LA-NF correlation (post-pre)',fontsize=40)
plt.ylabel('reduction in LA reactivity',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()