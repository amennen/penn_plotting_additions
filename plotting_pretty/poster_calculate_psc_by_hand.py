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
import matplotlib.pyplot as plt
font = {'weight' : 'normal',
        'size'   : 22}
import pandas as pd
import seaborn as sns
matplotlib.rc('font', **font)
def convertTR(timing):    
    TR = np.floor(timing/2)
    TRint = int(TR)
    return TRint

fmriprep_out="/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/fmriprep"
task_path = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/faces'
#save_path = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/normalized_runs_baseline'
save_path = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/highpass_normalized_runs_baseline'
amygdala_mask = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/mni_anat/LAMYG_in_MNI_overlapping.nii.gz'
timing_path = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/timing_files';
analyses_out = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti/derivatives/afni/first_level/stats'


all_categories = ['fearful','happy', 'neutral', 'object']

# BEFORE YOU DO ANYTHING SMOOTH THE DATA!
#subjectNum = np.int(sys.argv[1])
subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101, 102,103,104,105,106, 107,108,109,110,111,112,113,114])
allsubjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101,102,103,104,105,106,107,108,109,110,111,112,113,114])
HC_ind = np.argwhere(allsubjects<100)[:,0]
MDD_ind = np.argwhere(allsubjects>100)[:,0]
subjectNum=1
print(bids_id)
sessions = [1,3]
nRuns = 2
mask = amygdala_mask
trial=0
ntrials=6
# first just get negative
nsub = len(subjects)
# do ts of start + 24 s (12 TRs)
nTR = 12
ndays = len(sessions)
negative_ts = np.zeros((nsub,nTR,ndays))
neutral_ts = np.zeros((nsub,nTR,ndays))

for s in np.arange(nsub):
    subjectNum = subjects[s]
    d=0
    bids_id = 'sub-{0:03d}'.format(subjectNum)
    print(bids_id)
    for ses in sessions:
        subjectDay = ses
        ses_id = 'ses-{0:02d}'.format(subjectDay)
        print(ses_id)
        day_path=os.path.join(save_path,bids_id,ses_id)

            # now load in timing (and convert to TR #)
        all_start_timing = np.zeros((len(all_categories),3,nRuns))
        for c in np.arange(len(all_categories)):
            category = all_categories[c]
            category_str = category + '.txt'
            file_name = os.path.join(timing_path,bids_id,ses_id, category_str)
            t = pd.read_fwf(file_name,header=None)
            timing = t.values # now 2 x 18 array
            all_start_timing[c,:,0] = np.array([convertTR(timing[0,trial]),convertTR(timing[0,trial]+ntrials),convertTR(timing[0,trial+(ntrials*2)])])
            all_start_timing[c,:,1] = np.array([convertTR(timing[1,trial]),convertTR(timing[1,trial]+ntrials),convertTR(timing[1,trial+(ntrials*2)])])
        run_response = np.zeros((nRuns,nTR))
        run_response_neutral = np.zeros((nRuns,nTR))
        for r in np.arange(nRuns):
            faces_nifti_fn = glob.glob(os.path.join(day_path,'*task-faces_rec-uncorrected_run-0{0}_bold_space-MNI*preproc*'.format(r+1)))
            faces_nifti = faces_nifti_fn[0]
            # now load in with mask
            masked_img = nilearn.masking.apply_mask(faces_nifti,amygdala_mask)

            ROI_act = np.nanmean(masked_img,axis=1)
            # get run activity for the 3 times repeated each time
            block1Start = int(all_start_timing[0,0,r])
            block2Start = int(all_start_timing[0,1,r])
            block3Start = int(all_start_timing[0,2,r])
            block1 = ROI_act[block1Start:block1Start+nTR]
            block2 = ROI_act[block2Start:block2Start+nTR]
            block3 = ROI_act[block3Start:block3Start+nTR]
            run_response[r,:] = np.nanmean(np.concatenate((block1[:,np.newaxis],block2[:,np.newaxis],block3[:,np.newaxis]),axis=1),axis=1)


            block1Start = int(all_start_timing[2,0,r])
            block2Start = int(all_start_timing[2,1,r])
            block3Start = int(all_start_timing[2,2,r])
            block1 = ROI_act[block1Start:block1Start+nTR]
            block2 = ROI_act[block2Start:block2Start+nTR]
            block3 = ROI_act[block3Start:block3Start+nTR]
            run_response_neutral[r,:] = np.nanmean(np.concatenate((block1[:,np.newaxis],block2[:,np.newaxis],block3[:,np.newaxis]),axis=1),axis=1)
        negative_ts[s,:,d] = np.nanmean(run_response,axis=0)
        neutral_ts[s,:,d] = np.nanmean(run_response_neutral,axis=0)
            # plt.plot(ROI_act)
            # plt.show()
        d+=1


colors_dark = ['#636363','#de2d26']
colors_light = ['#636363','#de2d26']
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

plt.legend()
plt.show()



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
