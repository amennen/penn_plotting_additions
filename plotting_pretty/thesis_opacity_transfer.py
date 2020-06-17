# purpose: plot transfer function for opacity


import os
import glob
import argparse
import numpy as np  # type: ignore
import sys
from sklearn.metrics import roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
import scipy
import csv
sys.path.append(os.getcwd())
font = {'size': 22,
        'weight': 'normal'}
matplotlib.rc('font', **font)

from anne_additions.plotting_pretty.commonPlotting import *

gain=2.3
x_shift=.2
y_shift=.12
steepness=.9

x_vals = np.linspace(-1,1,100)
y_vals = steepness/(1+np.exp(-gain*(x_vals-x_shift)))+y_shift
#bins = [-1.   , -0.975, -0.9, -0.8 ,-0.7,-0.55,-0.4,-0.2,0,0.2,0.4,0.55,0.7, 0.8 ,  0.9 , 0.975, 1. ]

fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
plt.plot(x_vals,y_vals,color='k',lw=5)
plt.xlabel('scene - face classification', fontsize=25)
plt.ylabel('scene opacity', fontsize=25)
plt.title('Transfer function', fontsize=30)
plt.yticks(np.array([0,0.25,0.5,0.75,1]),fontsize=20,weight='normal')
plt.xticks(np.array([-1,-.5,0,0.5,1]),fontsize=20,weight='normal')
plt.plot([-2,2],[y_vals[0],y_vals[0]], '--', lw=1, color='k')
plt.plot([-2,2],[y_vals[-1],y_vals[-1]], '--', lw=1, color='k')
ax.tick_params(axis='x', which='major', pad=10)
print(y_vals[0])
print('****')
print(y_vals[-1])
# 0.17357192937885135
# ****
# 0.8966538366820863
#plt.plot([-2,2],[np.min(y_vals),np.min(y_vals)],'-',color='r',lw=3)
#plt.plot([-2,2],[np.max(y_vals),np.max(y_vals)],'-',color='r',lw=3)
# for b in np.arange(len(bins)):
#     # plot dashed for each bin
#     plt.plot([bins[b],bins[b]],[0,1],'--',color='b')
plt.plot(x_vals, y_vals, color='k', lw=3)
plt.ylim([0,1])
plt.xlim([-1,1])
plt.savefig('thesis_plots_checked/transferfn.pdf')
#plt.show()


# now check inverse

#x_vals2 = (1/gain)*(-1)*np.log((steepness/(y_vals-y_shift)) - 1) + x_shift