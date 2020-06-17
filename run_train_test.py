# Purpose--call train test frunction

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
from sklearn.linear_model import LogisticRegression
from rtfMRI.StructDict import StructDict, MatlabStructDict
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import scipy
from rtAtten.Test_L2_RLR_realtime import Test_L2_RLR_realtime
from sklearn import metrics
from anne_additions.getcs2 import train_test_python_classifier


subjectNum=sys.argv[1]

train_test_python_classifier(subjectNum)
