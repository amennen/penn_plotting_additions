"""Purpose: compute transitions in states during neurofeedback """
# load data
# bin classification separation into states
# plot

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

def getMADRSscoresALL():
    """Purpose: read MADRS.csv file and organize into a dictionary
    by subject number"""
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


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def transition_matrix(transitions,nstates):
    #n = 1+ max(transitions) #number of states
    n=nstates
    # initialize transition matrix M at 0
    M = [[0]*n for _ in range(n)]
    for (i,j) in zip(transitions,transitions[1:]):
        # if you transition from state i --> state j, add 1 in that spot
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        # get total number of times you were in state i
        s = sum(row)
        if s > 0:
            # divide by the total number of times for a probability
            row[:] = [f/s for f in row]
    return M

def transition_matrix_shift(transitions,nstates,nshift):
    n=nstates
    M = [[0]*n for _ in range(n)]
    # now we look a specified amount of time ahead - nshift
    for (i,j) in zip(transitions,transitions[nshift:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

