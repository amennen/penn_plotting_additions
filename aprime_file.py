"""
Functions to help process real-time fMRI data after-the-fact. Processes all the block data from a full run
"""


import numpy as np
import glob 
import sys
import os
import os
import glob
import argparse
import sys
# Add current working dir so main can be run from the top level rtAttenPenn directory
sys.path.append(os.getcwd())
import rtfMRI.utils as utils
import rtfMRI.ValidationUtils as vutils
from rtfMRI.RtfMRIClient import loadConfigFile
from rtfMRI.Errors import ValidationError
from rtAtten.RtAttenModel import getSubjectDayDir
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from rtfMRI.StructDict import StructDict, MatlabStructDict
from sklearn.metrics import roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
import csv
import matplotlib.pyplot as plt
import math

# NOTE: will have to un-wrap the sample data from the sampledata directory in order for this to work now
def get_blockData(subjNumb, day, run):
    data_files = glob.glob('/data/jux/cnds/amennen/rtAttenPenn/behavgonogo' + '/subject'+str(subjNumb)+'/day'+str(day)+'/run'+str(run)+'/blockdata_'+str(run)+'*.mat')
    if len(data_files)>0:
        filename = data_files[-1]
        behav = utils.loadMatFile(filename)
        data = behav['blockData']
    else:
        data = -1
    return data 
def get_blockData_realtime(subjNumb,day,run):
	data_files = glob.glob('/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo' + '/subject' + str(subjNumb) + '/day' + str(day) + '/run' + str(run) + '/blockdata_' + str(run) + '*.mat')
	print(data_files)
	filename = data_files[-1]
	behav = utils.loadMatFile(filename)
	data = behav['blockData']
	return data

def getImgProp(subjNumb,day,run):
    data = get_blockData_realtime(subjNumb,day,run)
    n_realtime_blocks = 4
    n_trs = 25
    all_opacity = np.zeros((n_realtime_blocks*n_trs,))
    for b in np.arange(n_realtime_blocks):
        sm_img_prop = np.mean(data['attImgProp'][:,b+4][0],axis=0)
        sm_img_prop[0:4] = np.nan
        all_opacity[b*n_trs:(b+1)*n_trs] = sm_img_prop
        # set the first 4 to nan 
    return all_opacity

def getSmoothImgProp(subjNumb,day,run):
    data = get_blockData_realtime(subjNumb,day,run)
    n_realtime_blocks = 4
    n_trs = 25
    all_opacity = np.zeros((n_realtime_blocks*n_trs,))
    for b in np.arange(n_realtime_blocks):
        sm_img_prop = np.mean(data['smoothAttImgProp'][:,b+4][0],axis=0)
        sm_img_prop[0:4] = np.nan
        all_opacity[b*n_trs:(b+1)*n_trs] = sm_img_prop
        # set the first 4 to nan 
    return all_opacity


def get_blockType(data):
	blockTypes = np.zeros((8,1))
	for b in np.arange(8):
		blockTypes[b] = data['specificBlock'][:,b][0][0][0]
	return blockTypes[:,0].astype(int)

def aprime(h,fa):
    if h + fa != 2 and h + fa != 0: 
    	if np.greater_equal(h,fa): a = .5 + (((h-fa) * (1+h-fa)) / (4 * h * (1-fa)))
    	else: a = .5 - (((fa-h) * (1+fa-h)) / (4 * fa * (1-h)))
    else:
        a = np.nan
    return a
# Takes in the 'blockdata' from a given run as input, and returns and 1x8 array indicating what the 'attended category'
# was for each of the 8 blocks in the run (1 = scene, 2 = face)
def get_attCategs(data):
    # what category are they attending to during this block? 1 = attending to face, 2 = attending to scene 
    # make a list of 8 values, which tells you the attended category for each block in the run 
    # so attCateg[i] is the category of what they are supposed to attend to for each of the 8 blocks in this 
    # run (i from 0-7)
    temp_attCateg = data.attCateg[0,:]
    attCateg = []
    attCateg[:] = [temp_attCateg[i][0] for i in range(0,len(temp_attCateg))]
    return attCateg


    

# Takes the 'blockdata' from a given run as input, and returns 2 things:
# (1) an 8x50 array of the scene category for each scene image (indoor or outdoor)
# (2) an 8x50 array of the face category for each face image (neutral male, neutral female, sad male, sad female,
#      happy male, happy female)
def get_imCategs(data):
    # create a data structure to store whether the scene is indoor(1)/outdoor(2) for each stimulus
    # sceneImCategs[i][j] gives the category of the scene (indoor or outdoor), with i from 0-7 (each block) and 
    # j from 0-49 (each trial)
    temp_sceneImCategs = data.categs[0,:]
    sceneImCategs = []
    sceneImCategs[:] = [temp_sceneImCategs[i][0][0][0] for i in range(0,len(temp_sceneImCategs))]


    # create a data structure to store whether the face is male/female and happy/neutral/sad for each stimulus
    # neutral M = 3, neutral F = 4, sad M = 5, sad F = 6, happy M = 7, happy F = 8
    # faceImCategs[i][j] gives the category of the face , with i from 0-7 (each block) and 
    # j from 0-49 (each trial)
    temp_faceImCategs = data.categs[0,:]
    faceImCategs = []
    faceImCategs[:] = [temp_faceImCategs[i][0][1][0] for i in range(0,len(temp_faceImCategs))]

    return sceneImCategs, faceImCategs



# Takes the 'blockdata' from a given run as input, and returns an 8x50 array indicating whether each trial was
# a Go Trial (1) or a No-Go Trial (0), as well as an 8x50 array indicating whether the subject pressed (1) or didn't 
# press (0) at each trial
def get_trialTypes_and_responses(data):
    # create a data structure to store the trial type (Go or No-Go) for each trial. 
    # trialType[i][j] gives the trial type for trial j in block i (i from 0-7, j from 0-49)
    temp_trialType = data.corrresps[0,:]
    trialType = []

    trialType[:] = [temp_trialType[i][0] for i in range(0, len(temp_trialType))]
    flatList_trialType = [item for sublist in trialType for item in sublist]

    temp2_trialType = flatList_trialType
    temp2_trialType = [0  if np.isnan(i) else 1.0 for i in flatList_trialType]
    trialType = np.reshape(temp2_trialType, [8,50])
    
    # create a data structure to store their behavioral responses (press vs no-press)
    # 1 = press, 0 = no press. responses[i][j] gives subject's response on trial j (0-49) during block i (0-7)
    temp_responses = data.accs[0,:]
    responses = []

    responses[:] = [temp_responses[i][0] for i in range(0, len(temp_responses))]
    flatList_responses = [item for sublist in responses for item in sublist]
   
    temp2_responses = flatList_responses 
    temp2_responses = [1 if flatList_responses[i] == 0 and temp2_trialType[i] == 0 else flatList_responses[i] for i in
                       range(0,len(flatList_responses))]
    temp2_responses = [0 if i == 2 else i for i in temp2_responses]
    responses = np.reshape(temp2_responses, [8,50])
    
    return trialType, responses

def getReactionTimes(data):
    rts = data.rts[0]
    # want simple matrix nblocks x ntrials
    rt_matrix = np.zeros((8,50))
    for b in np.arange(8):
        rt_matrix[b,:] = rts[b][0,:]
    return rt_matrix

def getRT_CR_FA(lureTrials,CR_trials,FA_trials,block_rts):
    """block is 1-index so subtract 1!!! - this is for 3 trials before lure, lure, 3 trials after lure"""
    nlures = len(lureTrials)
    ntrials = 50
    nFAs = len(FA_trials[0])
    if nFAs < 1:
        rt_FA = np.zeros((1,7)) * np.nan
    else:
        rt_FA = np.zeros((nFAs,7)) * np.nan
    nCRs = len(CR_trials[0])
    if nCRs < 1:
        rt_CR = np.zeros((1,7)) * np.nan
    else:
        rt_CR = np.zeros((nCRs,7)) * np.nan
    i_CR = 0
    i_FA = 0
    for l in np.arange(nlures):
        this_trial = lureTrials[l]
        # actually just get both bc will be nan no matter what if they don't press
        start = this_trial - 3
        stop = this_trial + 3 # include this trial
        if start < 0:
            start = 0
        if stop > ntrials - 1:
            stop = ntrials - 1
        n_points = (stop-start) + 1

        for shift in np.arange(n_points):
            this_specific_index = start + shift
            index_shift = this_specific_index - this_trial + 3
            # we want -3 --> 0 index, 0 --> 3, 3 --> 6
            if this_trial in FA_trials[0]:
                # incorrect false alarm
                rt_FA[i_FA,index_shift] = block_rts[this_specific_index]
                pressed = 1
            elif this_trial in CR_trials[0]:
                # correct correct rejection
                # look 3 trials before, 3 trials after
                rt_CR[i_CR,index_shift] = block_rts[this_specific_index]
                pressed = 0
        if pressed:
            i_FA += 1
        else:
            i_CR +=1
    block_rt_CR = np.nanmean(rt_CR,axis=0)
    block_rt_FA = np.nanmean(rt_FA,axis=0)
    return block_rt_CR, block_rt_FA

def get_decMatrices_and_aPrimes(data,realtime):
    trialType, responses = get_trialTypes_and_responses(data)
    rt_matrix = getReactionTimes(data)
    # for each block, we want reaction times for correct rejections and false alarms
    # for each lure, look at all RTs 3 before, trial, 3 after
    # at each correct rejection, we want RTs for 3 trials before and 3 trials after lure
    # at each false alarm, we want RTs from -3 to +3
    if not realtime:
        specificTypes = get_blockType(data)
    else:
        specificTypes = -1 # we know what the categories are
    block1_trialType = trialType[0]
    block1_responses = responses[0]

    block2_trialType = trialType[1]
    block2_responses = responses[1]

    block3_trialType = trialType[2]
    block3_responses = responses[2]

    block4_trialType = trialType[3]
    block4_responses = responses[3]

    block5_trialType = trialType[4]
    block5_responses = responses[4]

    block6_trialType = trialType[5]
    block6_responses = responses[5]

    block7_trialType = trialType[6]
    block7_responses = responses[6]

    block8_trialType = trialType[7]
    block8_responses = responses[7]

    run_hitRates = [] # store all the hit rates for the blocks in this run (1st val is hit rate for block 1, etc.)
    run_missRates = []
    run_FAs = []
    run_CRs = []
    run_aprimes = []
    run_rt_CR = np.zeros((8,7)) * np.nan
    run_rt_FA = np.zeros((8,7)) * np.nan
    # get general average RTs for each category
    avg_RT_FA = np.zeros((8,)) * np.nan
    avg_RT_HIT = np.zeros((8,))
    # code: Hit = 1, Miss = -1, False Alarm = -2, Correct Rejection = 2
    # block 1
    block1_results = np.zeros(len(trialType[0]))
    for i in range(0,len(block1_results)):
        if (block1_trialType[i] == 1 and block1_responses[i] == 1): # hit 
            block1_results[i] = 1
        elif (block1_trialType[i] == 1 and block1_responses[i] == 0): # miss
            block1_results[i] = -1
        elif (block1_trialType[i] == 0 and block1_responses[i] == 1): # false alarm
            block1_results[i] = -2
        elif (block1_trialType[i] == 0 and block1_responses[i] == 0): # correct rejection 
            block1_results[i] = 2

    block1_hits = np.where(block1_results == 1)
    block1_misses = np.where(block1_results == -1)
    block1_FAs = np.where(block1_results == -2)
    block1_CRs = np.where(block1_results == 2)
    block1_hit_rate = len(block1_hits[0])/(len(block1_hits[0]) + len(block1_misses[0]))
    run_hitRates.append(block1_hit_rate)
    block1_miss_rate = len(block1_misses[0])/(len(block1_hits[0]) + len(block1_misses[0]))
    run_missRates.append(block1_miss_rate)
    block1_FA_rate = len(block1_FAs[0])/(len(block1_FAs[0]) + len(block1_CRs[0]))
    run_FAs.append(block1_FA_rate)
    block1_CR_rate = len(block1_CRs[0])/(len(block1_FAs[0]) + len(block1_CRs[0]))
    run_CRs.append(block1_CR_rate)

    block1_aprime = aprime(block1_hit_rate, block1_FA_rate)
    run_aprimes.append(block1_aprime)
    block1_lure = np.where(block1_trialType == 0)[0]
    run_rt_CR[0,:],run_rt_FA[0,:] = getRT_CR_FA(block1_lure,block1_CRs,block1_FAs,rt_matrix[0,:])

    avg_RT_FA[0] = np.nanmean(rt_matrix[0,block1_FAs])
    avg_RT_HIT[0] = np.nanmean(rt_matrix[0,block1_hits])

    # block 2
    block2_results = np.zeros(len(trialType[0]))
    for i in range(0,len(block2_results)):
        if (block2_trialType[i] == 1 and block2_responses[i] == 1): # hit 
            block2_results[i] = 1
        elif (block2_trialType[i] == 1 and block2_responses[i] == 0): # miss
            block2_results[i] = -1
        elif (block2_trialType[i] == 0 and block2_responses[i] == 1): # false alarm
            block2_results[i] = -2
        elif (block2_trialType[i] == 0 and block2_responses[i] == 0): # correct rejection 
            block2_results[i] = 2

    block2_hits = np.where(block2_results == 1)
    block2_misses = np.where(block2_results == -1)
    block2_FAs = np.where(block2_results == -2)
    block2_CRs = np.where(block2_results == 2)
    block2_hit_rate = len(block2_hits[0])/(len(block2_hits[0]) + len(block2_misses[0]))
    run_hitRates.append(block2_hit_rate)
    block2_miss_rate = len(block2_misses[0])/(len(block2_hits[0]) + len(block2_misses[0]))
    run_missRates.append(block2_miss_rate)
    block2_FA_rate = len(block2_FAs[0])/(len(block2_FAs[0]) + len(block2_CRs[0]))
    run_FAs.append(block2_FA_rate)
    block2_CR_rate = len(block2_CRs[0])/(len(block2_FAs[0]) + len(block2_CRs[0]))
    run_CRs.append(block2_CR_rate)

    block2_aprime = aprime(block2_hit_rate, block2_FA_rate)
    run_aprimes.append(block2_aprime)
    block2_lure = np.where(block2_trialType == 0)[0]
    run_rt_CR[1,:],run_rt_FA[1,:] = getRT_CR_FA(block2_lure,block2_CRs,block2_FAs,rt_matrix[1,:])
    avg_RT_FA[1] = np.nanmean(rt_matrix[1,block2_FAs])
    avg_RT_HIT[1] = np.nanmean(rt_matrix[1,block2_hits])
    # block 3
    block3_results = np.zeros(len(trialType[0]))
    for i in range(0,len(block3_results)):
        if (block3_trialType[i] == 1 and block3_responses[i] == 1): # hit 
            block3_results[i] = 1
        elif (block3_trialType[i] == 1 and block3_responses[i] == 0): # miss
            block3_results[i] = -1
        elif (block3_trialType[i] == 0 and block3_responses[i] == 1): # false alarm
            block3_results[i] = -2
        elif (block3_trialType[i] == 0 and block3_responses[i] == 0): # correct rejection 
            block3_results[i] = 2

    block3_hits = np.where(block3_results == 1)
    block3_misses = np.where(block3_results == -1)
    block3_FAs = np.where(block3_results == -2)
    block3_CRs = np.where(block3_results == 2)
    block3_hit_rate = len(block3_hits[0])/(len(block3_hits[0]) + len(block3_misses[0]))
    run_hitRates.append(block3_hit_rate)
    block3_miss_rate = len(block3_misses[0])/(len(block3_hits[0]) + len(block3_misses[0]))
    run_missRates.append(block3_miss_rate)
    block3_FA_rate = len(block3_FAs[0])/(len(block3_FAs[0]) + len(block3_CRs[0]))
    run_FAs.append(block3_FA_rate)
    block3_CR_rate = len(block3_CRs[0])/(len(block3_FAs[0]) + len(block3_CRs[0]))
    run_CRs.append(block3_CR_rate)

    block3_aprime = aprime(block3_hit_rate, block3_FA_rate)
    run_aprimes.append(block3_aprime)
    block3_lure = np.where(block3_trialType == 0)[0]
    run_rt_CR[2,:],run_rt_FA[2,:] = getRT_CR_FA(block3_lure,block3_CRs,block3_FAs,rt_matrix[2,:])
    avg_RT_FA[2] = np.nanmean(rt_matrix[2,block3_FAs])
    avg_RT_HIT[2] = np.nanmean(rt_matrix[2,block3_hits])
    # block 4
    block4_results = np.zeros(len(trialType[0]))
    for i in range(0,len(block4_results)):
        if (block4_trialType[i] == 1 and block4_responses[i] == 1): # hit 
            block4_results[i] = 1
        elif (block4_trialType[i] == 1 and block4_responses[i] == 0): # miss
            block4_results[i] = -1
        elif (block4_trialType[i] == 0 and block4_responses[i] == 1): # false alarm
            block4_results[i] = -2
        elif (block4_trialType[i] == 0 and block4_responses[i] == 0): # correct rejection 
            block4_results[i] = 2

    block4_hits = np.where(block4_results == 1)
    block4_misses = np.where(block4_results == -1)
    block4_FAs = np.where(block4_results == -2)
    block4_CRs = np.where(block4_results == 2)
    block4_hit_rate = len(block4_hits[0])/(len(block4_hits[0]) + len(block4_misses[0]))
    run_hitRates.append(block4_hit_rate)
    block4_miss_rate = len(block4_misses[0])/(len(block4_hits[0]) + len(block4_misses[0]))
    run_missRates.append(block4_miss_rate)
    block4_FA_rate = len(block4_FAs[0])/(len(block4_FAs[0]) + len(block4_CRs[0]))
    run_FAs.append(block4_FA_rate)
    block4_CR_rate = len(block4_CRs[0])/(len(block4_FAs[0]) + len(block4_CRs[0]))
    run_CRs.append(block4_CR_rate)

    block4_aprime = aprime(block4_hit_rate, block4_FA_rate)
    run_aprimes.append(block4_aprime)
    block4_lure = np.where(block4_trialType == 0)[0]
    run_rt_CR[3,:],run_rt_FA[3,:] = getRT_CR_FA(block4_lure,block4_CRs,block4_FAs,rt_matrix[3,:])
    avg_RT_FA[3] = np.nanmean(rt_matrix[3,block4_FAs])
    avg_RT_HIT[3] = np.nanmean(rt_matrix[3,block4_hits])
    # block 5
    block5_results = np.zeros(len(trialType[0]))
    for i in range(0,len(block5_results)):
        if (block5_trialType[i] == 1 and block5_responses[i] == 1): # hit 
            block5_results[i] = 1
        elif (block5_trialType[i] == 1 and block5_responses[i] == 0): # miss
            block5_results[i] = -1
        elif (block5_trialType[i] == 0 and block5_responses[i] == 1): # false alarm
            block5_results[i] = -2
        elif (block5_trialType[i] == 0 and block5_responses[i] == 0): # correct rejection 
            block5_results[i] = 2

    block5_hits = np.where(block5_results == 1)
    block5_misses = np.where(block5_results == -1)
    block5_FAs = np.where(block5_results == -2)
    block5_CRs = np.where(block5_results == 2)
    block5_hit_rate = len(block5_hits[0])/(len(block5_hits[0]) + len(block5_misses[0]))
    run_hitRates.append(block5_hit_rate)
    block5_miss_rate = len(block5_misses[0])/(len(block5_hits[0]) + len(block5_misses[0]))
    run_missRates.append(block5_miss_rate)
    block5_FA_rate = len(block5_FAs[0])/(len(block5_FAs[0]) + len(block5_CRs[0]))
    run_FAs.append(block5_FA_rate)
    block5_CR_rate = len(block5_CRs[0])/(len(block5_FAs[0]) + len(block5_CRs[0]))
    run_CRs.append(block5_CR_rate)

    block5_aprime = aprime(block5_hit_rate, block5_FA_rate)
    run_aprimes.append(block5_aprime)
    block5_lure = np.where(block5_trialType == 0)[0]
    run_rt_CR[4,:],run_rt_FA[4,:] = getRT_CR_FA(block5_lure,block5_CRs,block5_FAs,rt_matrix[4,:])
    avg_RT_FA[4] = np.nanmean(rt_matrix[4,block5_FAs])
    avg_RT_HIT[4] = np.nanmean(rt_matrix[4,block5_hits])


    # block 6
    block6_results = np.zeros(len(trialType[0]))
    for i in range(0,len(block6_results)):
        if (block6_trialType[i] == 1 and block6_responses[i] == 1): # hit 
            block6_results[i] = 1
        elif (block6_trialType[i] == 1 and block6_responses[i] == 0): # miss
            block6_results[i] = -1
        elif (block6_trialType[i] == 0 and block6_responses[i] == 1): # false alarm
            block6_results[i] = -2
        elif (block6_trialType[i] == 0 and block6_responses[i] == 0): # correct rejection 
            block6_results[i] = 2

    block6_hits = np.where(block6_results == 1)
    block6_misses = np.where(block6_results == -1)
    block6_FAs = np.where(block6_results == -2)
    block6_CRs = np.where(block6_results == 2)
    block6_hit_rate = len(block6_hits[0])/(len(block6_hits[0]) + len(block6_misses[0]))
    run_hitRates.append(block6_hit_rate)
    block6_miss_rate = len(block6_misses[0])/(len(block6_hits[0]) + len(block6_misses[0]))
    run_missRates.append(block6_miss_rate)
    block6_FA_rate = len(block6_FAs[0])/(len(block6_FAs[0]) + len(block6_CRs[0]))
    run_FAs.append(block6_FA_rate)
    block6_CR_rate = len(block6_CRs[0])/(len(block6_FAs[0]) + len(block6_CRs[0]))
    run_CRs.append(block6_CR_rate)

    block6_aprime = aprime(block6_hit_rate, block6_FA_rate)
    run_aprimes.append(block6_aprime)
    block6_lure = np.where(block6_trialType == 0)[0]
    run_rt_CR[5,:],run_rt_FA[5,:] = getRT_CR_FA(block6_lure,block6_CRs,block6_FAs,rt_matrix[5,:])
    avg_RT_FA[5] = np.nanmean(rt_matrix[5,block6_FAs])
    avg_RT_HIT[5] = np.nanmean(rt_matrix[5,block6_hits])


    # block 7
    block7_results = np.zeros(len(trialType[0]))
    for i in range(0,len(block7_results)):
        if (block7_trialType[i] == 1 and block7_responses[i] == 1): # hit 
            block7_results[i] = 1
        elif (block7_trialType[i] == 1 and block7_responses[i] == 0): # miss
            block7_results[i] = -1
        elif (block7_trialType[i] == 0 and block7_responses[i] == 1): # false alarm
            block7_results[i] = -2
        elif (block7_trialType[i] == 0 and block7_responses[i] == 0): # correct rejection 
            block7_results[i] = 2

    block7_hits = np.where(block7_results == 1)
    block7_misses = np.where(block7_results == -1)
    block7_FAs = np.where(block7_results == -2)
    block7_CRs = np.where(block7_results == 2)
    block7_hit_rate = len(block7_hits[0])/(len(block7_hits[0]) + len(block7_misses[0]))
    run_hitRates.append(block7_hit_rate)
    block7_miss_rate = len(block7_misses[0])/(len(block7_hits[0]) + len(block7_misses[0]))
    run_missRates.append(block7_miss_rate)
    block7_FA_rate = len(block7_FAs[0])/(len(block7_FAs[0]) + len(block7_CRs[0]))
    run_FAs.append(block7_FA_rate)
    block7_CR_rate = len(block7_CRs[0])/(len(block7_FAs[0]) + len(block7_CRs[0]))
    run_CRs.append(block7_CR_rate)

    block7_aprime = aprime(block7_hit_rate, block7_FA_rate)
    run_aprimes.append(block7_aprime)
    block7_lure = np.where(block7_trialType == 0)[0]
    run_rt_CR[6,:],run_rt_FA[6,:] = getRT_CR_FA(block7_lure,block7_CRs,block7_FAs,rt_matrix[6,:])
    avg_RT_FA[6] = np.nanmean(rt_matrix[6,block7_FAs])
    avg_RT_HIT[6] = np.nanmean(rt_matrix[6,block7_hits])

    # block 8
    block8_results = np.zeros(len(trialType[0]))
    for i in range(0,len(block8_results)):
        if (block8_trialType[i] == 1 and block8_responses[i] == 1): # hit 
            block8_results[i] = 1
        elif (block8_trialType[i] == 1 and block8_responses[i] == 0): # miss
            block8_results[i] = -1
        elif (block8_trialType[i] == 0 and block8_responses[i] == 1): # false alarm
            block8_results[i] = -2
        elif (block8_trialType[i] == 0 and block8_responses[i] == 0): # correct rejection 
            block8_results[i] = 2

    block8_hits = np.where(block8_results == 1)
    block8_misses = np.where(block8_results == -1)
    block8_FAs = np.where(block8_results == -2)
    block8_CRs = np.where(block8_results == 2)
    block8_hit_rate = len(block8_hits[0])/(len(block8_hits[0]) + len(block8_misses[0]))
    run_hitRates.append(block8_hit_rate)
    block8_miss_rate = len(block8_misses[0])/(len(block8_hits[0]) + len(block8_misses[0]))
    run_missRates.append(block8_miss_rate)
    block8_FA_rate = len(block8_FAs[0])/(len(block8_FAs[0]) + len(block8_CRs[0]))
    run_FAs.append(block8_FA_rate)
    block8_CR_rate = len(block8_CRs[0])/(len(block8_FAs[0]) + len(block8_CRs[0]))
    run_CRs.append(block8_CR_rate)

    block8_aprime = aprime(block8_hit_rate, block8_FA_rate)
    run_aprimes.append(block8_aprime)
    block8_lure = np.where(block8_trialType == 0)[0]
    run_rt_CR[7,:],run_rt_FA[1,:] = getRT_CR_FA(block8_lure,block8_CRs,block8_FAs,rt_matrix[7,:])
    avg_RT_FA[7] = np.nanmean(rt_matrix[7,block8_FAs])
    avg_RT_HIT[7] = np.nanmean(rt_matrix[7,block8_hits])
    return run_hitRates, run_missRates, run_FAs, run_CRs, run_aprimes, specificTypes, run_rt_CR, run_rt_FA, avg_RT_FA, avg_RT_HIT






