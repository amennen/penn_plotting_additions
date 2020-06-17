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
from nilearn import plotting,masking
import matplotlib.pyplot as plt
from anne_additions.plotting_pretty.commonPlotting import *

# purpose: load amygdala changes
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

fmriprep_out="/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/fmriprep"
trunc_save_dir = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/neurofeedback/trunc'
noise_save_dir = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/neurofeedback/clean'
confounds_dir = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/fsl/first_level/confound_EVs'
whole_brain_mask = '/data/jux/cnds/amennen/rtAttenPenn/MNI_things/mni_icbm152_t1_tal_nlin_asym_09c_BOLD_mask_Penn.nii'
rtAttenPath = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'
amygdala_mask = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/mni_anat/LAMYG_in_MNI_overlapping.nii.gz'


subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101, 102,103,104,105,106, 107,108,109,110,111,112,113,114])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
nsubs = len(subjects)
d1_runs = 6
d2_runs = 8
d3_runs = 7
d1_runs_keep = np.array([0,1,2])
d2_runs_keep = np.array([2,3,4])
d3_runs_keep = np.array([4,5,6])
sessions = [1,2,3]
bins = [-1.   , -0.975, -0.9, -0.8 ,-0.7,-0.55,-0.4,-0.2,0,0.2,0.4,0.55,0.7, 0.8 ,  0.9 , 0.975, 1. ]
nbins=len(bins)
nstates=nbins-1
all_subject_matrix = np.zeros((nsubs,nstates,nstates,8,len(sessions))) * np.nan
# saved all in third day 
subjectDay=3
ses_id = 'ses-{0:02d}'.format(subjectDay)

# each subject has their own matrix
for s in np.arange(nsubs):
	subjectNum = subjects[s]
	bids_id = 'sub-{0:03d}'.format(subjectNum)
	save_path = "{0}/{1}/{2}/amygdala_changes.npy".format(noise_save_dir,bids_id,ses_id)
	subj_matrix = np.load(save_path)
	all_subject_matrix[s,:,:,:,:] = subj_matrix

# now average matrix by first three runs day 0 and last three runs day 0 and compare differences across groups
# compute beginning, middle, and end
all_subject_matrix_avg_runs = np.zeros((nsubs,nstates,nstates,len(sessions)))*np.nan
data_day_1 = all_subject_matrix[:,:,:,d1_runs_keep,0]
all_subject_matrix_avg_runs[:,:,:,0] = np.nanmean(data_day_1,axis=3)
data_day_2 = all_subject_matrix[:,:,:,d2_runs_keep,1]
all_subject_matrix_avg_runs[:,:,:,1] = np.nanmean(data_day_2,axis=3)
data_day_3 = all_subject_matrix[:,:,:,d3_runs_keep,2]
all_subject_matrix_avg_runs[:,:,:,2] = np.nanmean(data_day_3,axis=3)

# now plot the differences across group 
labels_pos = np.array(bins).astype(np.float)
labels_pos = np.around(labels_pos,decimals=2).astype(np.str)
vmin=-3
vmax=3
#day 1 first plt.figure(figsize=(20,20))
d=2
# make sequential colormap
fig,ax = plt.subplots(figsize=(20,20))
plt.subplot(1,2,1)
this_plot_hc = np.nanmean(all_subject_matrix_avg_runs[HC_ind,:,:,d],axis=0)
plt.imshow(this_plot_hc,cmap='Reds',vmin=vmin,vmax=vmax,origin='lower')
#plt.colorbar()
plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('HC',fontsize=20)
plt.subplot(1,2,2)
this_plot_mdd = np.nanmean(all_subject_matrix_avg_runs[MDD_ind,:,:,d],axis=0)
plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.imshow(this_plot_mdd,cmap='Reds',vmin=vmin,vmax=vmax,origin='lower')
plt.xlabel('value B')
plt.title('MDD',fontsize=20)
#plt.colorbar()
plt.show()


# now do the same by day by group
this_plot_hc1 = np.nanmean(all_subject_matrix_avg_runs[HC_ind,:,:,0],axis=0)
this_plot_hc3 = np.nanmean(all_subject_matrix_avg_runs[HC_ind,:,:,2],axis=0)
this_plot_mdd1 = np.nanmean(all_subject_matrix_avg_runs[MDD_ind,:,:,0],axis=0)
this_plot_mdd3 = np.nanmean(all_subject_matrix_avg_runs[MDD_ind,:,:,2],axis=0)
plt.figure(figsize=(10,10))
plt.imshow(this_plot_hc3-this_plot_hc1,cmap='bwr',origin='lower') # for half, max diff is .16, for all days, max diff is 0.1
plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('HC2 - HC0')
plt.colorbar()
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(this_plot_mdd3-this_plot_mdd1,cmap='bwr',origin='lower') # for half, max diff is .11, for all days, max diff is 0.06
plt.yticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xticks(np.arange(nbins)-0.5,labels_pos,fontsize=8)
plt.xlabel('value B')
plt.ylabel('value A')
plt.title('MDD2 - MDD0')
plt.colorbar()
plt.show()

nTriangles = 5
nDays=len(sessions)
triangle_data = np.zeros((nsubs,nTriangles,nDays))
for s in np.arange(nsubs):
    for d in np.arange(nDays):
        ex = all_subject_matrix_avg_runs[s,:,:,d]
        diagonal = np.diagonal(ex)
        diagonal_m1 = np.diagonal(ex,offset=-1)
        diagonal_m2 = np.diagonal(ex,offset=-2)
        diagonal_m3 = np.diagonal(ex,offset=-3)
        triangle_data[s,0,d] = np.nanmean(np.concatenate([diagonal[0:3].flatten(),diagonal_m1[0:2].flatten(),diagonal_m2[0].flatten()]))
        triangle_data[s,1,d] = np.nanmean(np.concatenate([diagonal[3:6].flatten(),diagonal_m1[3:5].flatten(),diagonal_m2[3].flatten()]))
        triangle_data[s,2,d] = np.nanmean(np.concatenate([diagonal[6:10].flatten(),diagonal_m1[6:9].flatten(),diagonal_m2[6:8].flatten(),diagonal_m3[6].flatten()]))
        triangle_data[s,3,d] = np.nanmean(np.concatenate([diagonal[10:13].flatten(),diagonal_m1[10:12].flatten(),diagonal_m2[10].flatten()]))
        triangle_data[s,4,d] = np.nanmean(np.concatenate([diagonal[13:16].flatten(),diagonal_m1[13:15].flatten(),diagonal_m2[13].flatten()]))
# to do: figure out orientation and for what triangle goes with what, if indivudal differences lead to anything
fig = plotPosterStyle_multiplePTS(triangle_data,subjects)
plt.subplot(1,3,1)
plt.ylim([-1.5,1.5])
#plt.yticks(np.linspace(0,.4,5),fontsize=15)
plt.xlabel('triangle group')
x,y = nonNan(triangle_data[MDD_ind,0,0],triangle_data[HC_ind,0,0])
t,p = scipy.stats.ttest_ind(x,y)
#addSingleStat(p/2,0,np.nanmax(triangle_data),0.01)
plt.subplot(1,3,2)
plt.ylim([-1.5,1.5])
#plt.yticks([])
plt.xlabel('triangle group')
plt.subplot(1,3,3)
plt.ylim([-1.5,1.5])
#plt.yticks([])
plt.xlabel('triangle group')
plt.show()

bins = [-1.   , -0.975, -0.9, -0.8 ,-0.7,-0.55,-0.4,-0.2,0,0.2,0.4,0.55,0.7, 0.8 ,  0.9 , 0.975, 1. ]
# calculation: how does amygdala activity change when it goes from values A --> value B

# check that the squares are right here!!!
# square of neg to pos = 0:8, 8:
# square of most neg to most pos: 0:3, 13:, 0:5, 11:, 7, 8:
neg_to_pos = np.nanmean(np.nanmean(all_subject_matrix_avg_runs[:,0:4,12:,:],axis=1),axis=1)

stat = neg_to_pos
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
plt.ylabel('$\Delta(LA): neg \Rightarrow pos $')
plt.xticks(np.arange(2),('Early NF','Late NF',))

plt.show()


M = getMADRSscoresALL()
d1,d2,d3 = getMADRSdiff(M,subjects)

data_mat = stat
#all_neg_change = data_mat[:,0]
all_neg_change = data_mat[:,2] - data_mat[:,0]
colors = ['k', 'r'] # HC, MDD
colors = ['#636363','#de2d26']
#fig = plt.figure(figsize=(10,7))
fig,ax = plt.subplots(figsize=(12,10))
sns.despine()
for s in np.arange(nsubs):
  subjectNum  = subjects[s]
  madrs_change = d1[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(-1*all_neg_change[s],-1*d1[s],marker='.',ms=30,color=colors[style])
plt.xlabel('improvement in amygdala downregulation',fontsize=15)
plt.ylabel('improvement in MADRS',fontsize=20)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
x,y = nonNan(-1*all_neg_change,-1*d1)
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(1.5,8,text, ha='left',va='top',color='k',fontsize=20)
x,y = nonNan(-1*all_neg_change[MDD_ind],-1*d1[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(1.5,0,text, ha='left',va='top',color='k',fontsize=20)

plt.show()


pos_to_neg = np.nanmean(np.nanmean(all_subject_matrix_avg_runs[:,12:,0:4,:],axis=1),axis=1)
stat = pos_to_neg
fig,ax = plotPosterStyle_DF(stat[:,np.array([0,2])],subjects)
x,y = nonNan(stat[MDD_ind,0],stat[HC_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,.2,np.nanmax(stat),0.05,0,'$MDD_1 > HC_1$')
x,y = nonNan(stat[MDD_ind,2],stat[HC_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,np.nanmax(stat),0.05,0)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0.2,1.2,np.nanmax(stat)+.2,0.05,0,'$MDD_1 < MDD_3$')
plt.ylabel('$\Delta(LA): pos \Rightarrow neg $')
plt.xticks(np.arange(2),('Early NF','Late NF',))
plt.show()


data_mat = stat
#all_neg_change = data_mat[:,0]
all_neg_change = data_mat[:,2] - data_mat[:,0]
colors = ['k', 'r'] # HC, MDD
colors = ['#636363','#de2d26']
#fig = plt.figure(figsize=(10,7))
fig,ax = plt.subplots(figsize=(12,10))
sns.despine()
for s in np.arange(nsubs):
  subjectNum  = subjects[s]
  madrs_change = d1[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(all_neg_change[s],-1*d1[s],marker='.',ms=30,color=colors[style])
plt.xlabel('worsening of amygdala upregulation when seeing face after scene',fontsize=15)
plt.ylabel('improvement in MADRS',fontsize=20)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
x,y = nonNan(-1*all_neg_change,-1*d1)
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(1.5,8,text, ha='left',va='top',color='k',fontsize=20)
x,y = nonNan(all_neg_change[MDD_ind],-1*d1[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(1.5,0,text, ha='left',va='top',color='k',fontsize=20)

plt.show()



x,y = nonNan(all_neg_change[MDD_ind],-1*d1[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)




amyg_change = diff_stat[:,1] - diff_stat[:,0]
fig,ax = plt.subplots(figsize=(12,10))
sns.despine()
for s in np.arange(nsubs):
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


nf_change = all_matrices[0,0,:,:]
nf_by_day = nf_change[:,2] - nf_change[:,0]
fig,ax = plt.subplots(figsize=(12,10))
sns.despine()
for s in np.arange(nsubs):
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

x,y = nonNan(all_neg_change,-1*nf_by_day)
r,p=scipy.stats.pearsonr(x,y)

x,y = nonNan(all_neg_change[MDD_ind],-1*nf_by_day[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)