# purpose: plot transfer function for opacity


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
matplotlib.rc('font', **font)

gain=2.3
x_shift=.2
y_shift=.12
steepness=.9

x_vals = np.linspace(-1,1,100)
y_vals = steepness/(1+np.exp(-gain*(x_vals-x_shift)))+y_shift
bins = [-1.   , -0.975, -0.9, -0.8 ,-0.7,-0.55,-0.4,-0.2,0,0.2,0.4,0.55,0.7, 0.8 ,  0.9 , 0.975, 1. ]

plt.figure()
plt.plot(x_vals,y_vals,color='k',lw=5)
plt.xlabel('scene - face difference')
plt.ylabel('scene opacity level')
plt.plot([-2,2],[np.min(y_vals),np.min(y_vals)],'-',color='r',lw=3)
plt.plot([-2,2],[np.max(y_vals),np.max(y_vals)],'-',color='r',lw=3)
for b in np.arange(len(bins)):
    # plot dashed for each bin
    plt.plot([bins[b],bins[b]],[0,1],'--',color='b')
plt.ylim([0,1])
plt.xlim([-1,1])
plt.show()


# now check inverse

x_vals2 = (1/gain)*(-1)*np.log((steepness/(y_vals-y_shift)) - 1) + x_shift