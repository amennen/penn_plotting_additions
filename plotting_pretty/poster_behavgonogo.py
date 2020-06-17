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
font = {'weight' : 'normal',
        'size'   : 22}
import pandas as pd
import seaborn as sns
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
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels
from statsmodels.stats.anova import AnovaRM
from anne_additions.plotting_pretty.commonPlotting import *

import matplotlib.pyplot as plt
import math

#ATTSCENE_DISTNEUTFACE = 1;
#ATTSCENE_DISTSADFACE = 2;
#ATTNEUTFACE_DISTSCENE = 3;
#ATTHAPPYFACE_DISTSCENE = 4;

# to compare behav bias (aprime sad - aprime neutral) / aprime neutral?
realtime = 0
ndays = 4
nruns = 4
subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101,102,103,104,105,106,107,108,109,110,111,112,113,114])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
nsubs = len(subjects)
all_sadbias = np.zeros((nsubs,ndays))*np.nan
all_happybias = np.zeros((nsubs,ndays))*np.nan
all_neutralface = np.zeros((nsubs,ndays))*np.nan
all_neutralscene = np.zeros((nsubs,ndays))*np.nan
all_sadBlocks = np.zeros((nsubs,ndays))*np.nan
all_happyBlocks = np.zeros((nsubs,ndays))*np.nan
fa_sadBlocks = np.zeros((nsubs,ndays))*np.nan
fa_happyBlocks = np.zeros((nsubs,ndays))*np.nan
fa_attSceneBlocks = np.zeros((nsubs,ndays))*np.nan
fa_attFaceBlocks = np.zeros((nsubs,ndays))*np.nan
h_sadBlocks = np.zeros((nsubs,ndays))*np.nan
h_happyBlocks = np.zeros((nsubs,ndays))*np.nan
h_attSceneBlocks = np.zeros((nsubs,ndays))*np.nan
h_attFaceBlocks = np.zeros((nsubs,ndays))*np.nan
cr_attSceneBlocks = np.zeros((nsubs,ndays,7))*np.nan
cr_sadBlocks = np.zeros((nsubs,ndays,7))*np.nan
cr_attFaceBlocks = np.zeros((nsubs,ndays,7))*np.nan
cr_happyBlocks =np.zeros((nsubs,ndays,7))*np.nan
fa_attSceneBlocks =np.zeros((nsubs,ndays,7))*np.nan
fa_sadBlocks = np.zeros((nsubs,ndays,7))*np.nan
fa_attFaceBlocks = np.zeros((nsubs,ndays,7))*np.nan
fa_happyBlocks = np.zeros((nsubs,ndays,7))*np.nan
rt_FA = np.zeros((nsubs,ndays)) * np.nan

RTfa_sadBlocks = np.zeros((nsubs,ndays))*np.nan
RTfa_happyBlocks = np.zeros((nsubs,ndays))*np.nan
RTfa_attSceneBlocks = np.zeros((nsubs,ndays))*np.nan
RTfa_attFaceBlocks = np.zeros((nsubs,ndays))*np.nan
RTh_sadBlocks = np.zeros((nsubs,ndays))*np.nan
RTh_happyBlocks = np.zeros((nsubs,ndays))*np.nan
RTh_attSceneBlocks = np.zeros((nsubs,ndays))*np.nan
RTh_attFaceBlocks = np.zeros((nsubs,ndays))*np.nan

for s in np.arange(nsubs):
    subjectNum = subjects[s]
    for d in np.arange(ndays):
        print('s,d is %i,%i' % ((subjectNum,d)))
        subjectDay = d+1
        sadbias = np.zeros((nruns))
        happybias = np.zeros((nruns))
        neutralface = np.zeros((nruns))
        neutralscene = np.zeros((nruns))
        sadblocks = np.zeros((nruns))
        happyblocks = np.zeros((nruns))
        fa_sad = np.zeros((nruns))
        fa_happy = np.zeros((nruns))
        fa_attScene = np.zeros((nruns))
        fa_attFace = np.zeros((nruns))
        h_sad = np.zeros((nruns))
        h_happy = np.zeros((nruns))
        h_attScene = np.zeros((nruns))
        h_attFace = np.zeros((nruns))
        RTfa_sad = np.zeros((nruns))
        RTfa_happy = np.zeros((nruns))
        RTfa_attScene = np.zeros((nruns))
        RTfa_attFace = np.zeros((nruns))
        RTh_sad = np.zeros((nruns))
        RTh_happy = np.zeros((nruns))
        RTh_attScene = np.zeros((nruns))
        RTh_attFace = np.zeros((nruns))
        day_rt_CR = np.zeros((nruns,7,4)) * np.nan
        day_rt_FA = np.zeros((nruns,7,4)) * np.nan
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
                    run_hitRates, run_missRates, run_FAs, run_CRs, run_aprimes, specificTypes, run_rt_CR, run_rt_FA, avg_FA, avg_HIT = get_decMatrices_and_aPrimes(data,realtime)
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
                    sadblocks[r] = aprime_sad
                    happyblocks[r] = aprime_happy

                    fa_sad[r] = np.nanmean(np.array([run_FAs[sad_blocks[0]],run_FAs[sad_blocks[1]]]))
                    fa_happy[r] = np.nanmean(np.array([run_FAs[happy_blocks[0]],run_FAs[happy_blocks[1]]]))
                    fa_attScene[r] = np.nanmean(np.array([run_FAs[neut_distface_blocks[0]],run_FAs[neut_distface_blocks[1]]]))
                    fa_attFace[r] =  np.nanmean(np.array([run_FAs[neut_distscene_blocks[0]],run_FAs[neut_distscene_blocks[1]]]))


                    RTfa_sad[r] = np.nanmean(np.array([avg_FA[sad_blocks[0]],avg_FA[sad_blocks[1]]]))
                    RTfa_happy[r] = np.nanmean(np.array([avg_FA[happy_blocks[0]],avg_FA[happy_blocks[1]]]))
                    RTfa_attScene[r] = np.nanmean(np.array([avg_FA[neut_distface_blocks[0]],avg_FA[neut_distface_blocks[1]]]))
                    RTfa_attFace[r] =  np.nanmean(np.array([avg_FA[neut_distscene_blocks[0]],avg_FA[neut_distscene_blocks[1]]]))

                    h_sad[r] = np.nanmean(np.array([run_hitRates[sad_blocks[0]],run_hitRates[sad_blocks[1]]]))
                    h_happy[r] = np.nanmean(np.array([run_hitRates[happy_blocks[0]],run_hitRates[happy_blocks[1]]]))
                    h_attScene[r] = np.nanmean(np.array([run_hitRates[neut_distface_blocks[0]],run_hitRates[neut_distface_blocks[1]]]))
                    h_attFace[r] =  np.nanmean(np.array([run_hitRates[neut_distscene_blocks[0]],run_hitRates[neut_distscene_blocks[1]]]))

                    RTh_sad[r] = np.nanmean(np.array([avg_HIT[sad_blocks[0]],avg_HIT[sad_blocks[1]]]))
                    RTh_happy[r] = np.nanmean(np.array([avg_HIT[happy_blocks[0]],avg_HIT[happy_blocks[1]]]))
                    RTh_attScene[r] = np.nanmean(np.array([avg_HIT[neut_distface_blocks[0]],avg_HIT[neut_distface_blocks[1]]]))
                    RTh_attFace[r] =  np.nanmean(np.array([avg_HIT[neut_distscene_blocks[0]],avg_HIT[neut_distscene_blocks[1]]]))
                    day_rt_CR[r,:,0] = np.nanmean(run_rt_CR[neut_distface_blocks,:],axis=0)
                    day_rt_CR[r,:,1] = np.nanmean(run_rt_CR[sad_blocks,:],axis=0)
                    day_rt_CR[r,:,2] = np.nanmean(run_rt_CR[neut_distscene_blocks,:],axis=0)
                    day_rt_CR[r,:,3] = np.nanmean(run_rt_CR[happy_blocks,:],axis=0)

                    day_rt_FA[r,:,0] = np.nanmean(run_rt_FA[neut_distface_blocks,:],axis=0)
                    day_rt_FA[r,:,1] = np.nanmean(run_rt_FA[sad_blocks,:],axis=0)
                    day_rt_FA[r,:,2] = np.nanmean(run_rt_FA[neut_distscene_blocks,:],axis=0)
                    day_rt_FA[r,:,3] = np.nanmean(run_rt_FA[happy_blocks,:],axis=0)
            # specific block is the type of block the person gets
        all_sadbias[s,d] = np.nanmean(sadbias)
        all_happybias[s,d] = np.nanmean(happybias)
        all_neutralface[s,d] = np.nanmean(neutralface)
        all_neutralscene[s,d] = np.nanmean(neutralscene)
        all_happyBlocks[s,d] = np.nanmean(happyblocks)
        all_sadBlocks[s,d] = np.nanmean(sadblocks)

        fa_sadBlocks[s,d] = np.nanmean(fa_sad)
        fa_happyBlocks[s,d] = np.nanmean(fa_happy)
        fa_attSceneBlocks[s,d] = np.nanmean(fa_attScene)
        fa_attFaceBlocks[s,d] = np.nanmean(fa_attFace)

        h_sadBlocks[s,d] = np.nanmean(h_sad)
        h_happyBlocks[s,d] = np.nanmean(h_happy)
        h_attSceneBlocks[s,d] = np.nanmean(h_attScene)
        h_attFaceBlocks[s,d] = np.nanmean(h_attFace)


        RTfa_sadBlocks[s,d] = np.nanmean(RTfa_sad)
        RTfa_happyBlocks[s,d] = np.nanmean(RTfa_happy)
        RTfa_attSceneBlocks[s,d] = np.nanmean(RTfa_attScene)
        RTfa_attFaceBlocks[s,d] = np.nanmean(RTfa_attFace)

        RTh_sadBlocks[s,d] = np.nanmean(RTh_sad)
        RTh_happyBlocks[s,d] = np.nanmean(RTh_happy)
        RTh_attSceneBlocks[s,d] = np.nanmean(RTh_attScene)
        RTh_attFaceBlocks[s,d] = np.nanmean(RTh_attFace)

        cr_attSceneBlocks[s,d,:] = np.nanmean(day_rt_CR[:,:,0],axis=0) * 1000 # put in ms
        cr_sadBlocks[s,d,:] = np.nanmean(day_rt_CR[:,:,1],axis=0)* 1000
        cr_attFaceBlocks[s,d,:] = np.nanmean(day_rt_CR[:,:,2],axis=0)* 1000
        cr_happyBlocks[s,d,:] = np.nanmean(day_rt_CR[:,:,3],axis=0)* 1000
        fa_attSceneBlocks[s,d,:] = np.nanmean(day_rt_FA[:,:,0],axis=0)* 1000
        fa_sadBlocks[s,d,:] = np.nanmean(day_rt_FA[:,:,1],axis=0)* 1000
        fa_attFaceBlocks[s,d,:] = np.nanmean(day_rt_FA[:,:,2],axis=0)* 1000
        fa_happyBlocks[s,d,:] = np.nanmean(day_rt_FA[:,:,3],axis=0)* 1000
# check behavior over all tasks
# combine into matrix
#### NEW LOOK AT FA ONLY
diff = fa_sadBlocks - fa_attSceneBlocks
stat = fa_happyBlocks- fa_attFaceBlocks
stat = fa_sadBlocks 
fig = plotPosterStyle_DF(stat[:,np.array([0,2])],subjects)
plt.ylabel('FA - sad')
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(all_sadbias),0.03)
#plt.title('A''ignore_neutralF - A''ignore_sadF')
plt.show()

# now plot RTs
day=0
colors_dark = ['#636363','#de2d26']
colors_light = ['#636363','#de2d26']
x = np.arange(7)
y = cr_sadBlocks[HC_ind,day,:] 
fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC CR')
y = fa_sadBlocks[HC_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,ls='--',color=colors_dark[0], label='HC FA')
y = cr_sadBlocks[MDD_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD CR')
y = fa_sadBlocks[MDD_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,ls='--',color=colors_dark[1], label='MDD FA')
labels_pos_v = np.arange(-3,4)
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(np.arange(7),labels_pos,fontsize=30)
plt.xlabel('trial from lure')
plt.ylabel('RT (ms)')
plt.ylim([250,650])
plt.title('attend neutral scene; distract negative face')
plt.legend()
plt.show()

day=0
colors_dark = ['#636363','#de2d26']
colors_light = ['#636363','#de2d26']
fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
x = np.arange(7)
y = cr_attSceneBlocks[HC_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC CR')
y = fa_attSceneBlocks[HC_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,ls='--',color=colors_dark[0], label='HC FA')
y = cr_attSceneBlocks[MDD_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD CR')
y = fa_attSceneBlocks[MDD_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,ls='--',color=colors_dark[1], label='MDD FA')
labels_pos_v = np.arange(-3,4)
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(np.arange(7),labels_pos,fontsize=30)
plt.xlabel('trial from lure')
plt.ylabel('RT (ms)')
plt.ylim([250,650])
plt.title('attend neutral scene; distract neutral face')
plt.legend()
plt.show()

# subtract categories for both groups
cr_diff = cr_sadBlocks - cr_attSceneBlocks
fa_diff = fa_sadBlocks - fa_attSceneBlocks
day=2
fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
x = np.arange(7)
y = cr_diff[HC_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC CR')
y = fa_diff[HC_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,ls='--',color=colors_dark[0], label='HC FA')
y = cr_diff[MDD_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD CR')
y = fa_diff[MDD_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,ls='--',color=colors_dark[1], label='MDD FA')
labels_pos_v = np.arange(-3,4)
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(np.arange(7),labels_pos,fontsize=30)
plt.xlabel('trial from lure')
plt.ylabel('RT (ms)')
plt.ylim([-60,80])
plt.title('distract neg - distract neutral face')
plt.legend()
plt.show()
x1=fa_diff[HC_ind,day,3]
y1=fa_diff[MDD_ind,day,3]
x,y = nonNan(x1,y1)
t,p = scipy.stats.ttest_ind(x,y)

negative_bias_change = fa_diff[:,2,3] - fa_diff[:,0,3]

cr_diff = cr_happyBlocks - cr_attFaceBlocks
fa_diff = fa_happyBlocks - fa_attFaceBlocks
pos_bias_change = fa_diff[:,2,4] - fa_diff[:,0,4]
day=2
fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
x = np.arange(7)
y = cr_diff[HC_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC CR')
y = fa_diff[HC_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,ls='--',color=colors_dark[0], label='HC FA')
y = cr_diff[MDD_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD CR')
y = fa_diff[MDD_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,ls='--',color=colors_dark[1], label='MDD FA')
labels_pos_v = np.arange(-3,4)
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(np.arange(7),labels_pos,fontsize=30)
plt.xlabel('trial from lure')
plt.ylabel('RT (ms)')
plt.ylim([-80,80])
plt.title('attend happy - attend neutral face')
plt.legend()
plt.show()
x1=fa_diff[HC_ind,day,4]
y1=fa_diff[MDD_ind,day,4]
x,y = nonNan(x1,y1)
t,p = scipy.stats.ttest_ind(x,y)




day=0
colors_dark = ['#636363','#de2d26']
colors_light = ['#636363','#de2d26']
fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
x = np.arange(7)
y = cr_attFaceBlocks[HC_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC CR')
y = fa_attFaceBlocks[HC_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,ls='--',color=colors_dark[0], label='HC FA')
y = cr_attFaceBlocks[MDD_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD CR')
y = fa_attFaceBlocks[MDD_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,ls='--',color=colors_dark[1], label='MDD FA')
response_diff_neg = negative_ts[:,16,:] - negative_ts[:,8,:]
plt.legend()
plt.xlabel('trial from lure')
plt.ylabel('RT (ms)')
plt.ylim([300,650])
plt.title('attend neutral face; distract neutral scene')
plt.show()

day=0
colors_dark = ['#636363','#de2d26']
colors_light = ['#636363','#de2d26']
fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
x = np.arange(7)
y = cr_attSceneBlocks[HC_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[0], label='HC CR')
y = fa_attSceneBlocks[HC_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,ls='--',color=colors_dark[0], label='HC FA')
y = cr_attSceneBlocks[MDD_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,color=colors_dark[1], label='MDD CR')
y = fa_attSceneBlocks[MDD_ind,day,:]
ym = np.nanmean(y,axis=0)
yerr = scipy.stats.sem(y,axis=0,nan_policy='omit')
plt.errorbar(x,ym,yerr=yerr,ls='--',color=colors_dark[1], label='MDD FA')
labels_pos_v = np.arange(-3,4)
labels_pos = labels_pos_v.astype(np.str)
plt.xticks(np.arange(7),labels_pos,fontsize=30)
plt.legend()
plt.xlabel('trial from lure')
plt.ylabel('RT (ms)')
plt.ylim([300,650])
plt.title('attend happy face; distract neutral scene')
plt.show()
######################################### LOOK AT ANY POSSIBLE RT DIFFERENCES ######################################### 

ncat=4
cat = ['sad', 'attS', 'attF', 'happy']
all_RTfa = np.zeros((nsubs,ndays,ncat))
for s in np.arange(nsubs):
    all_RTfa[s,:,0] = RTfa_sadBlocks[s,:]* 1000
    all_RTfa[s,:,1] = RTfa_attSceneBlocks[s,:]* 1000
    all_RTfa[s,:,2] = RTfa_attFaceBlocks[s,:]* 1000
    all_RTfa[s,:,3] = RTfa_happyBlocks[s,:]* 1000

all_RTh = np.zeros((nsubs,ndays,ncat))
for s in np.arange(nsubs):
    all_RTh[s,:,0] = RTh_sadBlocks[s,:]* 1000
    all_RTh[s,:,1] = RTh_attSceneBlocks[s,:]* 1000
    all_RTh[s,:,2] = RTh_attFaceBlocks[s,:]* 1000
    all_RTh[s,:,3] = RTh_happyBlocks[s,:]* 1000


h_fa_diff = all_RTh - all_RTfa


cat=['A:S\nD:-F', 'A:S\nD:F', 'A:F\nD:S', 'A:+F\nD:S']
fig,ax=plotPosterStyle_DF_valence_separateGroups(all_RTh[:,np.array([0,2,3])],all_RTfa[:,np.array([0,2,3])],subjects,'HC',cat,'RT (ms)',['hr', 'fa'])
x,y = nonNan(all_RTfa[HC_ind,0,0],all_RTh[HC_ind,0,0])
t,p = scipy.stats.ttest_rel(x,y)
# for MDD now
x,y = nonNan(all_RTfa[MDD_ind,0,0],all_RTh[MDD_ind,0,0])
t,p = scipy.stats.ttest_rel(x,y)


# then happy face blocks
x,y = nonNan(all_RTfa[HC_ind,0,3],all_RTh[HC_ind,0,3])
t,p = scipy.stats.ttest_ind(x,y)

# plt.subplot(1,3,1)
# addComparisonStat_SYM(p,-.2,.2,np.nanmax(all_RTfa),0.05,0,'$RT_h > RT_fa$')
plt.xticks(np.arange(4),cat)
plt.ylim([200,600])
plt.show()
fig,ax=plotPosterStyle_DF_valence_separateGroups(all_RTh[:,np.array([0,2,3])],all_RTfa[:,np.array([0,2,3])],subjects,'MDD',cat,'RT (ms)',['hr', 'fa'])
plt.xticks(np.arange(4),cat)
plt.ylim([200,600])
plt.show()


######################################### LOOK AT ANY POSSIBLE RT DIFFERENCES ######################################### 


#all_data=trial_average_total_viewing - SIZE - subjects x days x emotions
ncat=4
cat = ['sad', 'attS', 'attF', 'happy']
all_fa = np.zeros((nsubs,ndays,ncat))
for s in np.arange(nsubs):
    all_fa[s,:,0] = fa_sadBlocks[s,:]
    all_fa[s,:,1] = fa_attSceneBlocks[s,:]
    all_fa[s,:,2] = fa_attFaceBlocks[s,:]
    all_fa[s,:,3] = fa_happyBlocks[s,d]

all_h = np.zeros((nsubs,ndays,ncat))
for s in np.arange(nsubs):
    all_h[s,:,0] = h_sadBlocks[s,:]
    all_h[s,:,1] = h_attSceneBlocks[s,:]
    all_h[s,:,2] = h_attFaceBlocks[s,:]
    all_h[s,:,3] = h_happyBlocks[s,d]
fig,ax=plotPosterStyle_DF_valence(all_fa[:,np.array([0,2,3]),:],subjects,cat,'false alarm rate')
# plt.subplot(1,2,1)
# plt.xticks(np.arange(3),('Pre NF','Post NF', '1M FU'))
# plt.subplot(1,2,2)
# plt.xticks(np.arange(3),('Pre NF','Post NF', '1M FU'))
plt.show()

# now we want to change the plot so hit and false alarm rates are done for each group first separately
#
# do separate groups for this plot
cat=['A:S\nD:-F', 'A:S\nD:F', 'A:F\nD:S', 'A:+F\nD:S']
fig,ax=plotPosterStyle_DF_valence_separateGroups(all_h[:,np.array([0,2,3])],all_fa[:,np.array([0,2,3])],subjects,'HC',cat,'rate',['hr', 'fa'])
plt.xticks(np.arange(4),cat)
plt.show()
fig,ax=plotPosterStyle_DF_valence_separateGroups(all_h[:,np.array([0,2,3])],all_fa[:,np.array([0,2,3])],subjects,'MDD',cat,'rate',['hr', 'fa'])
plt.xticks(np.arange(4),cat)
plt.show()
plotPosterStyle_DF_valence_separateGroups(data1,data2,subjects,group,emo,ylabel,labels)

combined = np.concatenate((all_neutralscene[:,:,np.newaxis],all_neutralface[:,:,np.newaxis],all_happyBlocks[:,:,np.newaxis],all_sadBlocks[:,:,np.newaxis]),axis=2)
combined_avg = np.nanmean(combined,axis=2)
fig = plotPosterStyle(combined_avg,subjects)
x,y = nonNan(combined_avg[HC_ind,0],combined_avg[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(combined_avg),0.03)
x,y = nonNan(combined_avg[HC_ind,1],combined_avg[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(combined_avg),0.03)
x,y = nonNan(combined_avg[HC_ind,2],combined_avg[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,2,np.nanmax(combined_avg),0.03)
plt.ylabel('all a primes')
plt.xticks(np.arange(3),('Pre NF', 'Mid NF', 'Post NF'))
#plt.title('A''ignore_neutralF - A''ignore_sadF')
plt.show()




fig = plotPosterStyle(all_sadbias,subjects)
plt.ylabel('sad bias')
x,y = nonNan(all_sadbias[HC_ind,0],all_sadbias[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(all_sadbias),0.03)
#plt.title('A''ignore_neutralF - A''ignore_sadF')
plt.show()


np.nanmean(all_sadbias[HC_ind,:],axis=0)
np.nanmean(all_sadbias[MDD_ind,:],axis=0)
scipy.stats.ttest_ind(all_sadbias[HC_ind,0],all_sadbias[MDD_ind,0])

scipy.stats.ttest_rel(all_sadbias[MDD_ind,0],all_sadbias[MDD_ind,2])
x,y=nonNan(all_sadbias[HC_ind,0],all_sadbias[HC_ind,2])
scipy.stats.ttest_rel(x,y)



fig = plotPosterStyle(all_happybias,subjects)
plt.ylabel('happy bias')
x,y = nonNan(all_happybias[HC_ind,0],all_happybias[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(all_happybias),0.03)
x,y = nonNan(all_happybias[HC_ind,1],all_happybias[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(all_happybias),0.03)
x,y = nonNan(all_happybias[HC_ind,2],all_happybias[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,2,np.nanmax(all_happybias),0.03)
#plt.title('A''ignore_neutralF - A''ignore_sadF')
plt.show()
np.nanmean(all_sadbias[HC_ind,:],axis=0)
np.nanmean(all_sadbias[MDD_ind,:],axis=0)
scipy.stats.ttest_ind(all_happybias[HC_ind,0],all_happybias[MDD_ind,0])
x,y=nonNan(all_happybias[HC_ind,2],all_happybias[MDD_ind,2])
scipy.stats.ttest_ind(x,y)
x,y=nonNan(all_happybias[MDD_ind,0],all_happybias[MDD_ind,2])
scipy.stats.ttest_ind(x,y)
x,y=nonNan(all_happybias[HC_ind,0],all_happybias[HC_ind,2])
scipy.stats.ttest_ind(x,y)



fig = plotPosterStyle(all_sadBlocks,subjects)
plt.ylabel('Aprime distract sad')
plt.show()

fig = plotPosterStyle(all_happyBlocks,subjects)
plt.ylabel('Aprime attend happy')
plt.show()

fig = plotPosterStyle(all_neutralscene,subjects)
plt.ylabel('Aprime attend neutral scene')
plt.show()

fig = plotPosterStyle(all_neutralface,subjects)
plt.ylabel('Aprime attend neutral face')
plt.show()

sadbias = all_sadbias.flatten()
happybias = all_happybias.flatten()
day = np.tile(np.arange(ndays),nsubs)
subject = np.repeat(subjects,ndays)
groups = ['HC' if i in HC_ind else 'MDD' for i in np.arange(nsubs)]
groups = np.repeat(groups,ndays)
DATA = {}
DATA['sadbias'] = sadbias
DATA['happybias'] = happybias
DATA['day'] = day
DATA['subject'] = subject
DATA['groups'] = groups
df = pd.DataFrame.from_dict(DATA)

# different version for poster--divide by day
pal = dict(HC='k', MDD='r')
g = sns.FacetGrid(df,col='day',palette=pal)
g.map(plt.scatter,'groups','sadbias',color=['k'],alpha=0.3)
g.map(sns.pointplot,'groups','sadbias',palette=pal,ci=68,alpha=0.5,scale=1.5,errwidth=5)

#g = sns.factorplot(data=df,x='groups',y='sadbias',col='day',kind='bar',ci=68,palette=['k','r'],alpha=0.5)
#g.map(sns.factorplot,data=df,x='groups',y='sadbias',col='day',kind='swarm',palette=['k','r'],alpha=0.5)
g.add_legend()
plt.show()
#plt.ylabel('sad bias')

pal = dict(HC='k', MDD='r')
g = sns.FacetGrid(df,col='day',palette=pal)
g.map(plt.scatter,'groups','happybias',color=['k'],alpha=0.3)
g.map(sns.pointplot,'groups','happybias',palette=pal,ci=68,alpha=0.5,scale=1.5,errwidth=5)

#g = sns.factorplot(data=df,x='groups',y='sadbias',col='day',kind='bar',ci=68,palette=['k','r'],alpha=0.5)
#g.map(sns.factorplot,data=df,x='groups',y='sadbias',col='day',kind='swarm',palette=['k','r'],alpha=0.5)
g.add_legend()
plt.show()
#plt.ylabel('sad bias')


plt.figure()
sns.barplot(data=df,x='day',y='sadbias',hue='groups',ci=68,palette=['k','r'],alpha=0.5)
plt.title('POS=more distracted at sad than neutral F')
plt.ylabel('sad bias')
plt.show()
# not really a question of attention to faces though it's how it can block out faces 
scipy.stats.ttest_ind(all_sadbias[HC_ind,0],all_sadbias[MDD_ind,0]) # p = 0.33
plt.figure()
sns.barplot(data=df,x='day',y='happybias',hue='groups',ci=68,palette=['k','r'],alpha=0.5)
plt.title('POS=more attentive to neutral S than happy S')
plt.ylabel('anti-positive bias')
plt.show()
scipy.stats.ttest_ind(all_happybias[HC_ind,0],all_happybias[MDD_ind,0]) # p = 0.16 - almost different

# run anova?
model = ols('sadbias ~ groups*day',data=df).fit()
model.summary()

model = ols('happybias ~ groups*day',data=df).fit()
model.summary()


##### now do the same w/ differences
saddiff1 = all_sadbias[:,1] - all_sadbias[:,0]
saddiff2 = all_sadbias[:,2] - all_sadbias[:,0]
sad_diff = np.hstack((saddiff1,saddiff2))
happydiff1 = all_happybias[:,1] - all_happybias[:,0]
happydiff2 = all_happybias[:,2] - all_happybias[:,0]
happy_diff = np.hstack((happydiff1,happydiff2))
t_diff = np.repeat(np.array([1,2]),nsubs)
groups = ['HC' if i in HC_ind else 'MDD' for i in np.arange(nsubs)]
groups = np.tile(groups,2)
subject = np.tile(subjects,2)

DATA = {}
DATA['sad_diff'] = sad_diff
DATA['happy_diff'] = happy_diff
DATA['subject'] = subject
DATA['groups'] = groups
DATA['t_diff'] = t_diff
df = pd.DataFrame.from_dict(DATA)
plt.figure()
sns.barplot(data=df,x='t_diff',y='sad_diff',hue='groups',ci=68,palette=['k','r'],alpha=0.5)
plt.title('POS=more distracted at sad than neutral F')
plt.ylabel('sad bias')
plt.show()


plt.figure()
sns.barplot(data=df,x='t_diff',y='happy_diff',hue='groups',ci=68,palette=['k','r'],alpha=0.5)
plt.title('POS=more attentive to neutral S than happy ')
plt.ylabel('sad bias')
plt.show()

#####################################################
ONLY_negative_bias_change = fa_sadBlocks[:,2,3] - fa_sadBlocks[:,0,3]
M = getMADRSscoresALL()
d1,d2,d3 = getMADRSdiff(M,subjects)
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
for s in np.arange(nsubs):
  subjectNum  = subjects[s]
  madrs_change = d2[s]
  if subjectNum < 100:
    style = 0
  elif subjectNum > 100:
    style = 1
  plt.plot(-1*pos_bias_change[s],-1*negative_bias_change[s],marker='.',ms=30,color=colors[style])
#plt.xlabel('improvement in negative stickiness',fontsize=40)
plt.xlabel('improvement in negative RT bias',fontsize=20)
plt.ylabel('improvement in MADRS',fontsize=20)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim([-.5,.8])
plt.title('')
x,y = nonNan(-1*pos_bias_change,-1*negative_bias_change)
r,p=scipy.stats.pearsonr(x,y)
text='all subjects\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.12,15,text, ha='left',va='top',color='k',fontsize=25)
x,y = nonNan(-1*pos_bias_change[MDD_ind],-1*negative_bias_change[MDD_ind])
r,p=scipy.stats.pearsonr(x,y)
text='\nMDD only\nr = %2.2f\np = %2.2f' % (r,p)
plt.text(-.12,5,text, ha='left',va='top',color='k',fontsize=25)
# labels_pos_v = np.array([-0.4,0,0.4,0.8])
# labels_pos = labels_pos_v.astype(np.str)
# plt.xticks(labels_pos_v,labels_pos,fontsize=30)
#plt.savefig('poster_plots/MADRS_v_NF.png')
plt.show()