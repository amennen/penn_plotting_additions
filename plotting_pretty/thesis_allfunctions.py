# Purpose: make functions to calculate behavior for easy comparison with other tasks
# Written by: ACM
# 2/4/20


import numpy as np
import glob 
import sys
import os
import os
import scipy
import glob
import argparse
import sys
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
# Add current working dir so main can be run from the top level rtAttenPenn directory
sys.path.append(os.getcwd())
import rtfMRI.utils as utils
import rtfMRI.ValidationUtils as vutils
from rtfMRI.RtfMRIClient import loadConfigFile
from rtfMRI.Errors import ValidationError

import matplotlib
import matplotlib.pyplot as plt
font = {'size'   : 22}
import pandas as pd
import seaborn as sns
matplotlib.rc('font', **font)
import nilearn
from nilearn import image, masking
import csv
from anne_additions.aprime_file import aprime,get_blockType ,get_blockData, get_attCategs, get_imCategs, get_trialTypes_and_responses, get_decMatrices_and_aPrimes
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels
from statsmodels.stats.anova import AnovaRM
from anne_additions.plotting_pretty.commonPlotting import *

import matplotlib.pyplot as plt
import math

# new version 8/7/20: setting path as global variable
PROJECT_PATH =  '/cbica/projects/rtAtten/amennen/' # TODO: remove the amennen once done copying over
PROJECT_DATA_PATH = PROJECT_PATH + '/' + 'rtAttenPenn'

def getMADRSscoresALL():
  csv_fn = PROJECT_DATA_PATH  + '/' +  'MADRS.csv'
  nVisits = 4
  MADRS_SCORES = {}
  with open(csv_fn) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      line_count = 0
      for row in csv_reader:
          if len(row) > 0:
                  if 'RT' in row[0]:
                          subjectID = row[0]
                          subject_num = np.int(row[1])
                          goodrow=row
                          subject_scores = np.zeros((nVisits,))
                          subject_scores.fill(np.nan)
                          nInfo = len(goodrow)
                          for v in np.arange(2,nInfo):
                                  if len(goodrow[v]) > 0:
                                          subject_scores[v-2] = np.int(goodrow[v])
                          MADRS_SCORES[subject_num] = subject_scores
  return MADRS_SCORES

def getMADRSdiff(MADRS_SCORES,all_subjects):
  n_subs = len(all_subjects)
  diff_v5_v1 = np.zeros((n_subs,))
  diff_v6_v1 = np.zeros((n_subs,))
  diff_v7_v1 = np.zeros((n_subs,))

  for s in np.arange(n_subs):
    subject_num  = all_subjects[s]
    this_sub_madrs = MADRS_SCORES[subject_num]
    diff_v5_v1[s] = this_sub_madrs[1] - this_sub_madrs[0]
    diff_v6_v1[s] = this_sub_madrs[2] - this_sub_madrs[0]
    diff_v7_v1[s] = this_sub_madrs[3] - this_sub_madrs[0]
  return diff_v5_v1,diff_v6_v1,diff_v7_v1
  
def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def transition_matrix(transitions,n_states):
    #n = 1+ max(transitions) #number of states
    n=n_states
    M = [[0]*n for _ in range(n)]
    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M
    
def getAUC(subjects, take_half):
    n_days = 3
    n_subs = len(subjects)
    d1_runs = 6
    d2_runs = 8
    d3_runs = 7
    day_average = np.nan*np.zeros((n_subs,3))
    rtAttenPath = PROJECT_DATA_PATH  + '/' + 'fmridata/behavdata/gonogo'
    for s in np.arange(n_subs):
        subjectDir = rtAttenPath + '/' + 'subject' + str(subjects[s])
        outfile = subjectDir + '/' 'offlineAUC_RTCS.npz'    
        z=np.load(outfile)
        AUC = z['auc']
        if subjects[s] == 106:
            d1_runs = 5
        else:
            d1_runs = 6
        if take_half:
            d1_runs = 3
        day1avg = np.nanmean(AUC[0:d1_runs,0])
        if take_half:
            runs_taken = np.array([2,3,4])
        else:
            runs_taken = np.arange(d2_runs)
        day2avg = np.nanmean(AUC[runs_taken,1])
        if take_half:
            runs_taken = np.arange(4,d3_runs)
        else:
            runs_taken = np.arange(d3_runs)
        day3avg = np.nanmean(AUC[runs_taken,2])
        day_average[s,:] = np.array([day1avg,day2avg,day3avg])
    return day_average


def transition_matrix_shift(transitions,n_states,n_shift):
    n=n_states
    M = [[0]*n for _ in range(n)]
    for (i,j) in zip(transitions,transitions[n_shift:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

def getSpecificTypes(specific_types):
    """Return block types"""
    sad_blocks = np.argwhere(specific_types == 2)[:,0]
    happy_blocks = np.argwhere(specific_types == 4)[:,0]
    neut_distface_blocks = np.argwhere(specific_types == 1)[:,0]
    neut_distscene_blocks = np.argwhere(specific_types == 3)[:,0]
    return sad_blocks, happy_blocks, neut_distface_blocks, neut_distscene_blocks

def analyzeBehavGoNoGo(subjects):
    realtime = 0
    n_days = 4
    n_runs = 4
    HC_ind = np.argwhere(subjects<100)[:,0]
    MDD_ind = np.argwhere(subjects>100)[:,0]
    n_subs = len(subjects)
    all_sadbias = np.zeros((n_subs,n_days))*np.nan
    all_happybias = np.zeros((n_subs,n_days))*np.nan
    all_neutralface = np.zeros((n_subs,n_days))*np.nan
    all_neutralscene = np.zeros((n_subs,n_days))*np.nan
    all_sadBlocks = np.zeros((n_subs,n_days))*np.nan
    all_happyBlocks = np.zeros((n_subs,n_days))*np.nan
    fa_sadBlocks = np.zeros((n_subs,n_days))*np.nan
    fa_happyBlocks = np.zeros((n_subs,n_days))*np.nan
    fa_attSceneBlocks = np.zeros((n_subs,n_days))*np.nan
    fa_attFaceBlocks = np.zeros((n_subs,n_days))*np.nan
    h_sadBlocks = np.zeros((n_subs,n_days))*np.nan
    h_happyBlocks = np.zeros((n_subs,n_days))*np.nan
    h_attSceneBlocks = np.zeros((n_subs,n_days))*np.nan
    h_attFaceBlocks = np.zeros((n_subs,n_days))*np.nan
    cr_attSceneBlocks = np.zeros((n_subs,n_days,7))*np.nan
    cr_sadBlocks = np.zeros((n_subs,n_days,7))*np.nan
    cr_attFaceBlocks = np.zeros((n_subs,n_days,7))*np.nan
    cr_happyBlocks =np.zeros((n_subs,n_days,7))*np.nan
    fa_attSceneBlocks =np.zeros((n_subs,n_days,7))*np.nan
    fa_sadBlocks = np.zeros((n_subs,n_days,7))*np.nan
    fa_attFaceBlocks = np.zeros((n_subs,n_days,7))*np.nan
    fa_happyBlocks = np.zeros((n_subs,n_days,7))*np.nan
    rt_FA = np.zeros((n_subs,n_days)) * np.nan

    RTfa_sadBlocks = np.zeros((n_subs,n_days))*np.nan
    RTfa_happyBlocks = np.zeros((n_subs,n_days))*np.nan
    RTfa_attSceneBlocks = np.zeros((n_subs,n_days))*np.nan
    RTfa_attFaceBlocks = np.zeros((n_subs,n_days))*np.nan
    RTh_sadBlocks = np.zeros((n_subs,n_days))*np.nan
    RTh_happyBlocks = np.zeros((n_subs,n_days))*np.nan
    RTh_attSceneBlocks = np.zeros((n_subs,n_days))*np.nan
    RTh_attFaceBlocks = np.zeros((n_subs,n_days))*np.nan

    for s in np.arange(n_subs):
        subject_num = subjects[s]
        for d in np.arange(n_days):
            print('s,d is %i,%i' % ((subject_num,d)))
            subject_day = d+1
            sadbias = np.zeros((n_runs))*np.nan
            happybias = np.zeros((n_runs))*np.nan
            neutralface = np.zeros((n_runs))*np.nan
            neutralscene = np.zeros((n_runs))*np.nan
            sadblocks = np.zeros((n_runs))*np.nan
            happyblocks = np.zeros((n_runs))*np.nan
            fa_sad = np.zeros((n_runs))*np.nan
            fa_happy = np.zeros((n_runs))*np.nan
            fa_attScene = np.zeros((n_runs))*np.nan
            fa_attFace = np.zeros((n_runs))*np.nan
            h_sad = np.zeros((n_runs))*np.nan
            h_happy = np.zeros((n_runs))*np.nan
            h_attScene = np.zeros((n_runs))*np.nan
            h_attFace = np.zeros((n_runs))*np.nan
            RTfa_sad = np.zeros((n_runs))*np.nan
            RTfa_happy = np.zeros((n_runs))*np.nan
            RTfa_attScene = np.zeros((n_runs))*np.nan
            RTfa_attFace = np.zeros((n_runs))*np.nan
            RTh_sad = np.zeros((n_runs))*np.nan
            RTh_happy = np.zeros((n_runs))*np.nan
            RTh_attScene = np.zeros((n_runs))*np.nan
            RTh_attFace = np.zeros((n_runs))*np.nan
            day_rt_CR = np.zeros((n_runs,7,4)) * np.nan
            day_rt_FA = np.zeros((n_runs,7,4)) * np.nan
            for r in np.arange(4):
                run = r + 1
                # subject 108 didn't do 4 runs on day 2
                if subject_num == 108 and subject_day == 2 and run == 1:
                    pass
                else:
                    data = get_blockData(subject_num, subject_day, run)
                    if data != -1:
                        run_hitRates, run_missRates, run_FAs, run_CRs, run_aprimes, specificTypes, run_rt_CR, run_rt_FA, avg_FA, avg_HIT = get_decMatrices_and_aPrimes(data,realtime)
                        sad_blocks,happy_blocks,neut_distface_blocks,neut_distscene_blocks = getSpecificTypes(specificTypes)

                        aprime_sad = np.nanmean(np.array([run_aprimes[sad_blocks[0]],run_aprimes[sad_blocks[1]]]))
                        aprime_distneutface = np.nanmean(np.array([run_aprimes[neut_distface_blocks[0]],run_aprimes[neut_distface_blocks[1]]]))
                        aprime_happy = np.nanmean(np.array([run_aprimes[happy_blocks[0]],run_aprimes[happy_blocks[1]]]))
                        aprime_distneutscene = np.nanmean(np.array([run_aprimes[neut_distscene_blocks[0]],run_aprimes[neut_distscene_blocks[1]]]))

                        # aprime for each category
                        sadbias[r] = aprime_distneutface - aprime_sad # if sad is distracting would do worse so positive sad bias
                        happybias[r] = aprime_distneutscene - aprime_happy # if can't attend to happy would do worse so positive bias
                        neutralface[r] = aprime_distneutscene
                        neutralscene[r] = aprime_distneutface
                        sadblocks[r] = aprime_sad
                        happyblocks[r] = aprime_happy

                        # false alarm rates for each category
                        fa_sad[r] = np.nanmean(np.array([run_FAs[sad_blocks[0]],run_FAs[sad_blocks[1]]]))
                        fa_happy[r] = np.nanmean(np.array([run_FAs[happy_blocks[0]],run_FAs[happy_blocks[1]]]))
                        fa_attScene[r] = np.nanmean(np.array([run_FAs[neut_distface_blocks[0]],run_FAs[neut_distface_blocks[1]]]))
                        fa_attFace[r] =  np.nanmean(np.array([run_FAs[neut_distscene_blocks[0]],run_FAs[neut_distscene_blocks[1]]]))

                        # reaction time for each false alarm
                        RTfa_sad[r] = np.nanmean(np.array([avg_FA[sad_blocks[0]],avg_FA[sad_blocks[1]]]))
                        RTfa_happy[r] = np.nanmean(np.array([avg_FA[happy_blocks[0]],avg_FA[happy_blocks[1]]]))
                        RTfa_attScene[r] = np.nanmean(np.array([avg_FA[neut_distface_blocks[0]],avg_FA[neut_distface_blocks[1]]]))
                        RTfa_attFace[r] =  np.nanmean(np.array([avg_FA[neut_distscene_blocks[0]],avg_FA[neut_distscene_blocks[1]]]))

                        # hit rates for each category
                        h_sad[r] = np.nanmean(np.array([run_hitRates[sad_blocks[0]],run_hitRates[sad_blocks[1]]]))
                        h_happy[r] = np.nanmean(np.array([run_hitRates[happy_blocks[0]],run_hitRates[happy_blocks[1]]]))
                        h_attScene[r] = np.nanmean(np.array([run_hitRates[neut_distface_blocks[0]],run_hitRates[neut_distface_blocks[1]]]))
                        h_attFace[r] =  np.nanmean(np.array([run_hitRates[neut_distscene_blocks[0]],run_hitRates[neut_distscene_blocks[1]]]))

                        # reaction time for each hit
                        RTh_sad[r] = np.nanmean(np.array([avg_HIT[sad_blocks[0]],avg_HIT[sad_blocks[1]]]))
                        RTh_happy[r] = np.nanmean(np.array([avg_HIT[happy_blocks[0]],avg_HIT[happy_blocks[1]]]))
                        RTh_attScene[r] = np.nanmean(np.array([avg_HIT[neut_distface_blocks[0]],avg_HIT[neut_distface_blocks[1]]]))
                        RTh_attFace[r] =  np.nanmean(np.array([avg_HIT[neut_distscene_blocks[0]],avg_HIT[neut_distscene_blocks[1]]]))
                        
                        # correct rejections for each category
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

            # false alarm rates
            fa_sadBlocks[s,d] = np.nanmean(fa_sad)
            fa_happyBlocks[s,d] = np.nanmean(fa_happy)
            fa_attSceneBlocks[s,d] = np.nanmean(fa_attScene)
            fa_attFaceBlocks[s,d] = np.nanmean(fa_attFace)
            # hit rates
            h_sadBlocks[s,d] = np.nanmean(h_sad)
            h_happyBlocks[s,d] = np.nanmean(h_happy)
            h_attSceneBlocks[s,d] = np.nanmean(h_attScene)
            h_attFaceBlocks[s,d] = np.nanmean(h_attFace)

            # reaction times for false alarms
            RTfa_sadBlocks[s,d] = np.nanmean(RTfa_sad)
            RTfa_happyBlocks[s,d] = np.nanmean(RTfa_happy)
            RTfa_attSceneBlocks[s,d] = np.nanmean(RTfa_attScene)
            RTfa_attFaceBlocks[s,d] = np.nanmean(RTfa_attFace)

            # reaction times for hit
            RTh_sadBlocks[s,d] = np.nanmean(RTh_sad)
            RTh_happyBlocks[s,d] = np.nanmean(RTh_happy)
            RTh_attSceneBlocks[s,d] = np.nanmean(RTh_attScene)
            RTh_attFaceBlocks[s,d] = np.nanmean(RTh_attFace)

            # correct rejection-related RT analysis 
            cr_attSceneBlocks[s,d,:] = np.nanmean(day_rt_CR[:,:,0],axis=0) * 1000 # put in ms
            cr_sadBlocks[s,d,:] = np.nanmean(day_rt_CR[:,:,1],axis=0)* 1000
            cr_attFaceBlocks[s,d,:] = np.nanmean(day_rt_CR[:,:,2],axis=0)* 1000
            cr_happyBlocks[s,d,:] = np.nanmean(day_rt_CR[:,:,3],axis=0)* 1000
            fa_attSceneBlocks[s,d,:] = np.nanmean(day_rt_FA[:,:,0],axis=0)* 1000
            fa_sadBlocks[s,d,:] = np.nanmean(day_rt_FA[:,:,1],axis=0)* 1000
            fa_attFaceBlocks[s,d,:] = np.nanmean(day_rt_FA[:,:,2],axis=0)* 1000
            fa_happyBlocks[s,d,:] = np.nanmean(day_rt_FA[:,:,3],axis=0)* 1000
    return all_sadbias, all_happybias, all_neutralface, all_neutralscene, all_happyBlocks, all_sadBlocks

def consecutive(data,indices, step_size=1):
    fixations = np.split(data, np.where(np.diff(data) != step_size)[0]+1)
    if np.sum(indices) != -1:
      indices = np.split(indices, np.where(np.diff(data) != step_size)[0]+1)
    else:
      indices = np.nan
    return fixations,indices

def correlateROICS(subjects,take_half):
    rtAttenPath = PROJECT_DATA_PATH  + '/' + 'fmridata/behavdata/gonogo'
    n_days = 3
    d1_runs = 6
    d2_runs = 8
    d3_runs = 7
    n_subs = len(subjects)
    perception_correlation = np.zeros((n_subs,d2_runs,n_days))*np.nan
    attention_correlation = np.zeros((n_subs,d2_runs,n_days))*np.nan
    perception_attention_correlation = np.zeros((n_subs,d2_runs,n_days))*np.nan

    # loop through all subjects
    for s in np.arange(n_subs):
        subject_key = 'subject' + str(subjects[s])
        subject_dir = rtAttenPath + '/' + 'subject' + str(subjects[s])
        outfile = subject_dir + '/' + 'offlineAUC_RTCS.npz'  
        perception_outfile = subject_dir + '/' + 'offlineAUC_RTCS_perception.npz'   
        attention_outfile = subject_dir + '/' + 'offlineAUC_RTCS_attention.npz'  
        z=np.load(outfile)
        zp = np.load(perception_outfile)
        za = np.load(attention_outfile)
        if subjects[s] == 106:
            d1_runs = 5
        else:
            d1_runs = 6
        # load original category separation, 
        # load perception category separation
        # load attention category perception
        CS = z['csOverTime']
        CS_p = zp['csOverTime']
        CS_a = za['csOverTime']
        n_TR = np.shape(CS)[1]
        for d in np.arange(n_days):
          if d == 0:
            if not take_half:
              d1_take = np.arange(d1_runs)
              n_runs = d1_runs
            else:
              d1_take = np.arange(3)
              n_runs = 3
          elif d == 1:
            if not take_half:
              d2_take = np.arange(d2_runs)
              n_runs = d2_runs
            else:
              d2_take = np.array([2,3,4])
              n_runs=3
          elif d == 2:
            if not take_half:
              d3_take = np.arange(d3_runs)
              n_runs = d3_runs
            else:
              d3_take = np.arange(4,d3_runs)
              n_runs = 3
          
          categSep = CS[0:n_runs,:,d]
          categSep_p = CS_p[0:n_runs,:,d]
          categSep_a = CS_a[0:n_runs,:,d]
          for r in np.arange(n_runs):
            original = categSep[r,:]
            perception = categSep_p[r,:]
            attention = categSep_a[r,:]
            # correlate original with each network
            perception_correlation[s,r,d] = scipy.stats.pearsonr(original,perception)[0]
            attention_correlation[s,r,d] = scipy.stats.pearsonr(original,attention)[0]
            perception_attention_correlation[s,r,d] = scipy.stats.pearsonr(attention,perception)[0]
    return perception_correlation, attention_correlation, perception_attention_correlation


def buildMasterDict(subjects,take_half):
    all_x = {}
    all_opacity = {}
    rtAttenPath = PROJECT_DATA_PATH  + '/' + 'fmridata/behavdata/gonogo'
    master_x = []
    master_opacity=[]
    n_days = 3
    n_subs = len(subjects)
    d1_runs = 6
    d2_runs = 8
    d3_runs = 7
    for s in np.arange(n_subs):
        subject_key = 'subject' + str(subjects[s])
        all_x[subject_key] = {}
        all_opacity[subject_key] = {}
        subject_dir = rtAttenPath + '/' + 'subject' + str(subjects[s])
        outfile = subject_dir + '/' 'offlineAUC_RTCS.npz'    
        z=np.load(outfile)
        if subjects[s] == 106:
            d1_runs = 5
        else:
            d1_runs = 6
        CS = z['csOverTime']
        n_TR = np.shape(CS)[1]
        for d in np.arange(n_days):
            day_x = []
            day_opacity = []
            if d == 0:
                if take_half:
                    d1_runs = 3
                categSep = CS[0:d1_runs,:,0]
                n_runs = np.shape(categSep)[0]
            elif d == 1:
                if take_half:
                    runs_taken = np.array([2,3,4])
                else:
                    runs_taken = np.arange(d2_runs)
                categSep = CS[runs_taken,:,1]
                n_runs = np.shape(categSep)[0]
            elif d == 2:
                if take_half:
                    runs_taken = np.arange(4,d3_runs)
                else:
                    runs_taken = np.arange(d3_runs)
                categSep = CS[runs_taken,:,2]
                n_runs = np.shape(categSep)[0]
            vec = categSep.flatten()
            day_key = 'day' + str(d)
            day_x.append(vec)
            all_x[subject_key][day_key] = day_x
            master_x.extend(day_x[0])
    return master_x,all_x


def buildTransitionMatrix(subjects,all_x,bins,n_shift1,n_shift2):
    n_days = 3
    nbins=len(bins)
    n_subs = len(subjects)
    all_matrices = np.zeros((nbins-1,nbins-1,n_subs,n_days))
    p_state = np.zeros((n_subs,nbins-1,n_days))
    subject_averages = np.zeros((n_subs,n_days))
    combineShift=1

    for s in np.arange(n_subs):
        subject_key = 'subject' + str(subjects[s])
        for d in np.arange(n_days):
            day_key = 'day' + str(d)
            pos = all_x[subject_key][day_key][0]
            subject_averages[s,d] = np.mean(pos)
            indices = np.digitize(pos,bins)
            # if evidence is exactly 1.0 will be outside range, so scale back to 10
            #indices[np.argwhere(indices==len(pos_edges))] = len(pos_edges) - 1
            indices[np.argwhere(indices==len(bins))] = len(bins) - 1
            indices_0ind = indices.copy() - 1
            for st in np.arange(nbins-1):
                n_this_state = len(np.argwhere(indices_0ind==(st)))
                if n_this_state > 0:
                    p_state[s,st,d] = n_this_state/len(indices_0ind)
                else:
                    str_print = 'subject {0}, day {1}, never in state {2}'.format(subjects[s],d,st)
                    print(str_print)
            M1 = np.array(transition_matrix_shift(indices_0ind,nbins-1,n_shift1))
            if combineShift:
                M2 = np.array(transition_matrix_shift(indices_0ind,nbins-1,n_shift2))
                M_combined = np.concatenate((M1[:,:,np.newaxis],M2[:,:,np.newaxis]),axis=2)
                M = np.mean(M_combined,axis=2)
            else:
                M = M1
            #M,average_states = np.array(transition_matrix_shift_average_previous_states(indices_0ind,nbins-1,n_shift))
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
    return all_matrices,p_state,subject_averages


def getGazeData(subjects):
    # this is where the results for the fixations are stored after the matlab file
    # want: initial orientation, total ratio, first dwell time
    results = PROJECT_DATA_PATH  + '/' + 'gazedata/all_fixations.mat'
    HC_ind = np.argwhere(subjects<100)[:,0]
    MDD_ind = np.argwhere(subjects>100)[:,0]
    d = scipy.io.loadmat(results,struct_as_record=False)
    states = d['all_fixations'] # this is n_subjects x 12 trials x 3600 trials x 4 days
    #n_days = 3 # for now do 3 days --np.shape(ratios)[3]
    n_samples = np.shape(states)[2]
    n_trials = np.shape(states)[1]
    n_subjects = np.shape(states)[0]
    n_days = np.shape(states)[3]

    DYSPHORIC = 1;
    THREAT = 2;
    NEUTRAL = 3;
    POSITIVE = 4;
    emotions = ['DYSPHORIC', 'THREAT', 'NEUTRAL', 'POSITIVE']
    emo = ['DYS', 'THR', 'NEU', 'POS']

    nsamples_per_seg=600
    nsamples_per_sec=120
    n_emotions = len(emo)
    # calculate disengagement time: from start how long until you go until next image
    # would need to know index of all fixations starts
    fixation_durations_first_time =np.zeros((n_subjects,n_trials,n_days,n_emotions)) *np.nan
    n_fixations_per_trial = np.zeros((n_subjects,n_trials,n_days,n_emotions)) *np.nan
    first_orientation = np.zeros((n_subjects,n_trials,n_days,n_emotions)) *np.nan
    total_viewing_time = np.zeros((n_subjects,n_trials,n_days,n_emotions)) *np.nan
    options = np.arange(n_emotions)
    for s in np.arange(n_subjects):
      for d in np.arange(n_days):
        for t in np.arange(n_trials):
          all_trial_indices = np.arange(n_samples)
          trial_ts = states[s,t,:,d]
          fixations = trial_ts[trial_ts>0]
          fixations_trial_indices = all_trial_indices[trial_ts>0]
          if len(fixations): # looked somewhere during that trial
            # get first orientation
            orientation = fixations[0] - 1 # subtract 1 to account for matlab-python indexing
            other_options = options[options!=orientation] # get all the other ones that weren't the first orientation
            first_orientation[s,t,d,orientation] = 1 
            first_orientation[s,t,d,other_options] = 0

            n_points_recorded = len(fixations)
            # get consecutive parts across all trial
            c_trial,indices_trial = consecutive(fixations,fixations_trial_indices,step_size=0)
            n_fixations = len(c_trial)
            total_time_looking = n_points_recorded/(nsamples_per_sec*30)
            # calculate average fixation duration for that emotion each tmie
            # then set to zero
            # get total viewing time
            for e in np.arange(n_emotions):
                total_viewing_time[s,t,d,e] = len(np.argwhere(fixations==e+1))/n_points_recorded
            n_fixations_per_trial[s,t,d,:] = np.zeros((4,))
            for f in np.arange(n_fixations):
              fixationStart = indices_trial[f][0]

              if not np.any(np.diff(indices_trial[f])>1): # make sure this is actually the same fixation when we're on 1 image
                fixationStop=indices_trial[f][-1]
              else: # 2 separate views - not the same fixation even though it's the same image
                split_fix = np.argwhere(np.diff(indices_trial[f])>1)[0]
                fixationStop=indices_trial[f][split_fix]
              tdiff = (fixationStop-fixationStart)#/nsamples_per_sec
              # only record if the first one for that emotion
              emotion = c_trial[f][0] - 1 # this is the emotion of the current fixation
              n_fixations_per_trial[s,t,d,emotion] += 1 # add one more count to count how many times you fixated on that emotion
              # only put duration if first one
              if n_fixations_per_trial[s,t,d,emotion] == 1:
                fixation_durations_first_time[s,t,d,emotion] = (tdiff+1)/n_points_recorded #total_time_looking
                if fixation_durations_first_time[s,t,d,emotion] > 1: # this shouldn't be greater than 1 because it's a ratio - check
                  print(s,d,t,f)
            total_number_fixations = np.sum(n_fixations_per_trial[s,t,d,:])
            n_fixations_per_trial[s,t,d,:] = n_fixations_per_trial[s,t,d,:]/total_number_fixations

    return first_orientation,total_viewing_time,fixation_durations_first_time



def calculateAmygConnectivity(network_name,correlation_matrix,full_DF,system_DF,all_good_ROI):
    this_ROI = full_DF.ROI[system_DF==network_name].values.astype(int) 
    n_nodes = len(this_ROI)
    # now convert this to the indices
    amyg_indicies_in_matrix = np.where(np.in1d(all_good_ROI,265))[0]
    this_ROI_indices_in_matrix = np.where(np.in1d(all_good_ROI,this_ROI))[0]
    if network_name == 'Sensory/somatomotor Hand':
        # concatenate other one
        other_SMN = full_DF.ROI[system_DF=='Sensory/somatomotor Mouth'].values.astype(int) 
        other_SMN_indices_in_matrix = np.where(np.in1d(all_good_ROI,other_SMN))[0]
        this_ROI_indices_in_matrix = np.concatenate((this_ROI_indices_in_matrix,other_SMN_indices_in_matrix))
    corr_sum=0
    for i in this_ROI_indices_in_matrix:
        this_corr = correlation_matrix[i,amyg_indicies_in_matrix[0]]
        corr_sum = corr_sum + this_corr
    across_ROI_mean = (corr_sum)/n_nodes
    return across_ROI_mean

def calculateWithinConnectivity(network_name,correlation_matrix,full_DF,system_DF,all_good_ROI):
    # find DMN labels
    this_ROI = full_DF.ROI[system_DF==network_name].values.astype(int) 
    # now convert this to the indices
    this_ROI_indices_in_matrix = np.where(np.in1d(all_good_ROI,this_ROI))[0]
    if network_name == 'Sensory/somatomotor Hand':
        # concatenate other one
        other_SMN = full_DF.ROI[system_DF=='Sensory/somatomotor Mouth'].values.astype(int) 
        other_SMN_indices_in_matrix = np.where(np.in1d(all_good_ROI,other_SMN))[0]
        this_ROI_indices_in_matrix = np.concatenate((this_ROI_indices_in_matrix,other_SMN_indices_in_matrix))
    #x,y = np.meshgrid(this_ROI_indices_in_matrix,this_ROI_indices_in_matrix)
    #this_ROI_correlations = correlation_matrix[x,y]
    n_nodes = len(this_ROI)
    corr_sum = 0
    for i in this_ROI_indices_in_matrix:
        for j in this_ROI_indices_in_matrix:
            if i != j:
                this_corr =correlation_matrix[i,j]
                corr_sum = corr_sum + this_corr

    # within_ROI_sum = np.nansum(this_ROI_correlations)/2 # dividing by 2 because will be double the off-diagonal values
    within_ROI_mean = (corr_sum/2)/np.square(n_nodes)
    #within_ROI_mean = np.nanmean(this_ROI_correlations)/np.square(n_nodes)
    return within_ROI_mean


def calculatePairwiseConnectivity(network_A,network_B,correlation_matrix,full_DF,system_DF,all_good_ROI):
    A_ROI = full_DF.ROI[system_DF==network_A].values.astype(int) 
    A_ROI_indices_in_matrix = np.where(np.in1d(all_good_ROI,A_ROI))[0]
    if network_A == 'Sensory/somatomotor Hand':
        # concatenate other one
        other_SMN = full_DF.ROI[system_DF=='Sensory/somatomotor Mouth'].values.astype(int) 
        other_SMN_indices_in_matrix = np.where(np.in1d(all_good_ROI,other_SMN))[0]
        A_ROI_indices_in_matrix = np.concatenate((A_ROI_indices_in_matrix,other_SMN_indices_in_matrix))
    B_ROI = full_DF.ROI[system_DF==network_B].values.astype(int) 
    B_ROI_indices_in_matrix = np.where(np.in1d(all_good_ROI,B_ROI))[0]
    if network_B == 'Sensory/somatomotor Hand':
        # concatenate other one
        other_SMN = full_DF.ROI[system_DF=='Sensory/somatomotor Mouth'].values.astype(int) 
        other_SMN_indices_in_matrix = np.where(np.in1d(all_good_ROI,other_SMN))[0]
        B_ROI_indices_in_matrix = np.concatenate((B_ROI_indices_in_matrix,other_SMN_indices_in_matrix))
    x,y = np.meshgrid(A_ROI_indices_in_matrix,B_ROI_indices_in_matrix)
    # not dividing by 2 again because again ROI 1 is x and ROI 2 is y so we're not double counting anything
    across_ROI_correlations = correlation_matrix[x,y]
    n_nodes_A = len(A_ROI)
    n_nodes_B = len(B_ROI)
    across_ROI_sum = np.nansum(across_ROI_correlations)
    across_ROI_mean = across_ROI_sum/(n_nodes_A*n_nodes_B)
    # try another way
    corr_sum=0
    for i in A_ROI_indices_in_matrix:
        for j in B_ROI_indices_in_matrix:
            #print(i,j)
            this_corr = correlation_matrix[i,j]
            corr_sum = corr_sum + this_corr
    across_ROI_mean = this_corr/(n_nodes_A*n_nodes_B)
    return across_ROI_mean

def getFunctionalCon(subjects):
    powerAtlas = PROJECT_PATH + '/' + 'Power/power264MNI_resampled_amygdala.nii.gz'
    noise_save_dir = PROJECT_DATA_PATH + '/' + 'fmridata/Nifti/derivatives/resting/clean'
    amygdala_mask = PROJECT_DATA_PATH + '/' + 'fmridata/Nifti/derivatives/mni_anat/LAMYG_in_MNI_overlapping.nii.gz'

    nROI = 264
    labelsFile = PROJECT_PATH + '/' + 'Power/Neuron_consensus_264.csv'
    z = pd.read_csv(labelsFile)
    complete_labels=z[1:]
    ROI = complete_labels['ROI']
    system = complete_labels['Suggested System']
    all_systems = np.unique(system)
    systems_to_keep = ['Default mode','Fronto-parietal Task Control', 
                     'Visual','Subcortical', 'Cingulo-opercular Task Control',  'Salience', 'Ventral attention','Dorsal attention',
                     'Auditory','Sensory/somatomotor Hand', 'Sensory/somatomotor Mouth']
    # combine the two sennsory/somatomotor 
    n_systems = len(systems_to_keep) - 1
    # here we get the ROIs that have each of the labels we don't want
    # then we subtract 1 to go to python indices
    systems_to_remove = ['Uncertain', 'Cerebellar', 'Memory retrieval?']
    systems_to_keep_abbrv = ['DMN', 'FPN', 'VIS', 'SUB', 'CON', 'SAN', 'VAN', 'DAN', 'AUD','SMN']
    all_cer_labels = complete_labels.ROI[system=='Cerebellar'].values.astype(int) - 1
    all_mem_labels = complete_labels.ROI[system=='Memory retrieval?'].values.astype(int) - 1
    all_uncertain_labels = complete_labels.ROI[system=='Uncertain'].values.astype(int) - 1 # go from label to python index
    all_bad_labels = np.concatenate((all_cer_labels,all_mem_labels,all_uncertain_labels),axis=0)
    # left with 227 regions like beginning of Meichen's (removed the rest for bad signals)
    all_network_ind = np.arange(nROI)
    all_good_labels = [x for x in all_network_ind if x not in all_bad_labels]
    all_good_labels_amyg = all_good_labels + [264] # last index is amygdala
    all_good_ROI = np.array(all_good_labels_amyg) + 1 # puts as ROI labels so we can find the specific regions we want
    all_FPN_labels = complete_labels.ROI[system=='Fronto-parietal Task Control'].values.astype(int) - 1
    nROI_good = len(all_good_labels)
    n_sub = len(subjects)
    HC_ind = np.argwhere(subjects<100)[:,0]
    MDD_ind = np.argwhere(subjects>100)[:,0]
    sessions = [1,3]
    n_days = len(sessions)
    average_within_mat = np.zeros((n_systems,n_systems,n_sub,n_days))
    average_one_vs_all = np.zeros((n_systems,n_sub,n_days))
    amyg_con = np.zeros((n_sub,n_days))
    # NOW CALCULATE DATA FOR SUBJECTS
    for s in np.arange(n_sub):
        subject_num=subjects[s]
        bids_id = 'sub-{0:03d}'.format(subject_num)
        for ses in np.arange(n_days):
            subject_day=sessions[ses]
            ses_id = 'ses-{0:02d}'.format(subject_day)
            clean_path = noise_save_dir + '/' + bids_id + '/' + ses_id
            cleaned_image = '{0}/{1}_{2}_task_rest_glm.nii.gz'.format(clean_path,bids_id,ses_id)

            #cleaned_image_data = nib.load(cleaned_image).get_fdata()
            # doing standardize = True here at least makes it so voxels outside of brain would have 0 std and not be included
            masker = NiftiLabelsMasker(labels_img=powerAtlas, standardize=True,
                                       memory='nilearn_cache', verbose=5)
            time_series = masker.fit_transform(cleaned_image) # now data is n time points x 264 nodes
            time_series_good_labels = time_series[:,all_good_labels_amyg] # now data is in n time points x 227 nodes
            time_series_df = pd.DataFrame(time_series_good_labels)
            correlation_matrix = np.array(time_series_df.corr(method='pearson'))
            #correlation_measure = ConnectivityMeasure(kind='correlation')
            #correlation_matrix = correlation_measure.fit_transform([time_series_good_labels])[0] # takes correlation for all 227 nodes
            np.fill_diagonal(correlation_matrix,np.nan) # to make sure you don't get the same node in the within connectivity difference
            for row in np.arange(n_systems):
                for col in np.arange(n_systems):
                    if row == col: # diagonal
                        average_within_mat[row,col,s,ses] = calculateWithinConnectivity(systems_to_keep[row],correlation_matrix,complete_labels,system,all_good_ROI)
                    else:
                        average_within_mat[row,col,s,ses] = calculatePairwiseConnectivity(systems_to_keep[row],systems_to_keep[col],correlation_matrix,complete_labels,system,all_good_ROI)
                # now calculate oneVsAll
                #average_one_vs_all[row,s,ses] = calculateOneVsAllConnectivity(systems_to_keep[row],correlation_matrix,complete_labels,system,all_good_ROI)
            amyg_con[s,ses] = calculateAmygConnectivity(systems_to_keep[1],correlation_matrix,complete_labels,system,all_good_ROI)
    return average_within_mat, amyg_con

def convertTR(timing):    
    TR = np.floor(timing/2)
    TR_int = int(TR)
    return TR_int

def getRunResponse(category,run,all_start_timing,n_TR,ROI_act):
  """This is for the faces task """
  block1Start = int(all_start_timing[category,0,run])-5
  if block1Start < 0 * run:
    # pad with nans
    block1Start = 0
    block1 = np.concatenate((np.zeros(5,)*np.nan,ROI_act[block1Start:block1Start+n_TR-5]))
  else:
    block1 = ROI_act[block1Start:block1Start+n_TR]
  block2Start = int(all_start_timing[category,1,run])-5
  block3Start = int(all_start_timing[category,2,run])-5
  if block3Start+n_TR > 142*(run+1): # if this spills over to the end of the run, don't include
      block3 = np.concatenate((ROI_act[block3Start:block3Start+n_TR-5],np.zeros(5,)*np.nan))
  else:
      block3 = ROI_act[block3Start:block3Start+n_TR]
  block2 = ROI_act[block2Start:block2Start+n_TR]
  # take average over the whole response
  run_response = np.nanmean(np.concatenate((block1[:,np.newaxis],block2[:,np.newaxis],block3[:,np.newaxis]),axis=1),axis=1)
  return run_response

def getFacesBehav(subjects, ID_LIST):
    behavdata = PROJECT_DATA_PATH + '/' + 'fmridata/behavdata/faces'
    CONDITION=5
    RT=17
    ACC=16
    ndays=2

    HC_ind = np.argwhere(subjects<100)[:,0]
    MDD_ind = np.argwhere(subjects>100)[:,0]
    nsubjects = len(subjects)
    all_sadbias = np.zeros((nsubjects,ndays))
    all_happybias = np.zeros((nsubjects,ndays))
    all_acc = np.zeros((nsubjects,ndays))
    all_sadbias_acc = np.zeros((nsubjects,ndays))
    all_happybias_acc = np.zeros((nsubjects,ndays))
    for s in np.arange(nsubjects):
        ID = ID_LIST[s]
        for d in np.arange(ndays):
            day = d + 1
            happy_RT = []
            neutral_RT = []
            fear_RT = []
            happy_acc = []
            neutral_acc = []
            fear_acc = []
            sub_acc = []
            file_name = glob.glob(behavdata + '/' + ID + '/' + ID + '_Day' + str(day) + '_Scanner' + '*.csv')
            print(file_name[0])
            with open(file_name[0]) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if row[0] == 'A' or row[0] == 'B': # loop only through trial rows
                        this_condition = row[CONDITION]
                        this_RT = row[RT]
                        this_acc = row[ACC]
                        if this_condition == 'Happy':
                            if this_RT:
                                happy_RT.append(np.float64(this_RT))
                            happy_acc.append(np.float64(this_acc))
                        elif this_condition == 'Fearful':
                            if this_RT:
                                fear_RT.append(np.float64(this_RT))
                            fear_acc.append(np.float64(this_acc))
                        elif this_condition == 'Neutral':
                            if this_RT:
                                neutral_RT.append(np.float64(this_RT))
                            neutral_acc.append(np.float64(this_acc))
                        sub_acc.append(np.float64(this_acc))
            all_sadbias[s,d] = np.nanmean(neutral_RT) - np.nanmean(fear_RT)
            all_happybias[s,d] = np.nanmean(neutral_RT) - np.nanmean(happy_RT)
            all_sadbias_acc[s,d] = np.nanmean(neutral_acc) - np.nanmean(fear_acc)
            all_happybias_acc[s,d] = np.nanmean(neutral_acc) - np.nanmean(happy_acc)
            all_acc[s,d] = np.nanmean(sub_acc)
    return all_sadbias, all_happybias, all_acc, all_sadbias_acc, all_happybias_acc

def getFaces3dTProjectData(subjects,ROI):
    fmriprep_out = PROJECT_DATA_PATH + '/' + "fmridata/Nifti/derivatives/fmriprep"
    task_path = PROJECT_DATA_PATH + '/' + 'fmridata/behavdata/faces'
    save_path = PROJECT_DATA_PATH + '/' + 'fmridata/Nifti/derivatives/afni/first_level/highpass_normalized_runs_baseline'
    amygdala_mask = PROJECT_DATA_PATH + '/' + 'fmridata/Nifti/derivatives/mni_anat/LAMYG_in_MNI_overlapping.nii.gz'
    timing_path = PROJECT_DATA_PATH + '/' + 'fmridata/Nifti/derivatives/afni/first_level/timing_files';
    analyses_out = PROJECT_DATA_PATH + '/' + 'fmridata/Nifti/derivatives/afni/first_level/stats'
    ROI_DIR = PROJECT_DATA_PATH + '/' + 'MNI_things/clusters'
    # cluster indices 1& 2 is anterior cingulate  (dorsal & rostral)
    # cluster index 9 is for amygdalaj
    cluster = 9
    amyg_cluster = "{0}/cluster{1}sphere.nii.gz".format(ROI_DIR,cluster+1)
    all_categories = ['fearful','happy', 'neutral', 'object']
    dorsal_acc = "{0}/cluster{1}sphere.nii.gz".format(ROI_DIR,0+1)
    acc = "{0}/cluster1and2sphere.nii.gz".format(ROI_DIR)
    if ROI == 'amyg_overlapping':
        mask = amygdala_mask
    elif ROI == 'amyg_cluster':
        mask = amyg_cluster
    elif ROI == 'dorsal_acc':
        mask = dorsal_acc
    elif ROI == 'acc':
        mask = acc
    HC_ind = np.argwhere(subjects<100)[:,0]
    MDD_ind = np.argwhere(subjects>100)[:,0]
    sessions = [1,3]
    n_runs = 2
    trial=0
    n_trials=6
    # first just get negative
    n_sub = len(subjects)
    # do ts of start + 24 s (12 TRs)
    n_TR = 9+10 # theres's 18 seconds = 9 TRs so 9 + 10 (5 before, 5 after)
    n_days = len(sessions)
    negative_ts = np.zeros((n_sub,n_TR,n_days))
    neutral_ts = np.zeros((n_sub,n_TR,n_days))
    happy_ts = np.zeros((n_sub,n_TR,n_days))

    for s in np.arange(n_sub):
        subject_num = subjects[s]
        d=0
        bids_id = 'sub-{0:03d}'.format(subject_num)
        print(bids_id)
        for ses in sessions:
            subject_day = ses
            ses_id = 'ses-{0:02d}'.format(subject_day)
            print(ses_id)
            day_path=os.path.join(analyses_out,bids_id,ses_id)

            # now load in timing (and convert to TR #)
            all_start_timing = np.zeros((len(all_categories),3,n_runs))
            for c in np.arange(len(all_categories)):
                category = all_categories[c]
                category_str = category + '.txt'
                file_name = os.path.join(timing_path,bids_id,ses_id, category_str)
                t = pd.read_fwf(file_name,header=None)
                timing = t.values # now 2 x 18 array
                all_start_timing[c,:,0] = np.array([convertTR(timing[0,trial]),convertTR(timing[0,trial+n_trials]),convertTR(timing[0,trial+(n_trials*2)])])
                all_start_timing[c,:,1] = np.array([convertTR(timing[1,trial]),convertTR(timing[1,trial+n_trials]),convertTR(timing[1,trial+(n_trials*2)])])+142
            run_response_neg = np.zeros((n_runs,n_TR)) * np.nan
            run_response_neutral = np.zeros((n_runs,n_TR)) * np.nan
            run_response_happy = np.zeros((n_runs,n_TR)) * np.nan
            fn = glob.glob(os.path.join(day_path,'*_task-faces_glm_3dtproject.nii.gz'))
            output_img = fn[0]
            masked_img = nilearn.masking.apply_mask(output_img,mask)
            ROI_act = np.nanmean(masked_img,axis=1)

            # do for run 1 first
            for r in np.arange(n_runs):
              run_response_neg[r,:] = getRunResponse(0,r,all_start_timing,n_TR,ROI_act)
              run_response_neutral[r,:] = getRunResponse(2,r,all_start_timing,n_TR,ROI_act)
              run_response_happy[r,:] = getRunResponse(1,r,all_start_timing,n_TR,ROI_act)

            negative_ts[s,:,d] = np.nanmean(run_response_neg,axis=0)
            neutral_ts[s,:,d] = np.nanmean(run_response_neutral,axis=0)
            happy_ts[s,:,d] = np.nanmean(run_response_happy,axis=0)
            d+=1
    return negative_ts, neutral_ts, happy_ts, n_TR

