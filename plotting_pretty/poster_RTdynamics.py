# plot results/look at group differences

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
import matplotlib
import matplotlib.pyplot as plt
import scipy
font = {'weight' : 'normal',
        'size'   : 22}
import csv
from anne_additions.plotting_pretty.commonPlotting import *
matplotlib.rc('font', **font)
# for each subject, you need to run getcs.py in anne_additions first to get cs evidence for that subject
# have python and matlab versions--let's start with matlab 
def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101, 102,103,104,105,106, 107,108,109,110,111,112,113])
HC_ind = np.argwhere(subjects<100)[:,0]
MDD_ind = np.argwhere(subjects>100)[:,0]
nsubs = len(subjects)
d1_runs = 6
d2_runs = 8
d3_runs = 7
totalRuns = d1_runs + d2_runs + d3_runs

# now do the same thing for cs
day_average = np.zeros((nsubs,3))
day_average_neg = np.zeros((nsubs,3))
day_average_pos = np.zeros((nsubs,3))
cs_day_average = np.zeros((nsubs,3))

neg_dec = np.nan*np.zeros((nsubs,d2_runs,3))
neg_inc = np.nan*np.zeros((nsubs,d2_runs,3))
pos_dec = np.nan*np.zeros((nsubs,d2_runs,3))
pos_inc = np.nan*np.zeros((nsubs,d2_runs,3))
neg_area_all = np.nan*np.zeros((nsubs,d2_runs,3))
all_x = {}
all_dx = {}
rtAttenPath = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo'

# get HC averages for each RUN OF SCANNER/DAY
nDays = 3
for s in np.arange(nsubs):
#for s in np.arange(1):
#   s=0
    subject_key = 'subject' + str(subjects[s])
    all_x[subject_key] = {}
    all_dx[subject_key] = {}
    subjectDir = rtAttenPath + '/' + 'subject' + str(subjects[s])
    outfile = subjectDir + '/' 'offlineAUC_RTCS.npz'    
    z=np.load(outfile)
    if subjects[s] == 106:
        d1_runs = 5
    else:
        d1_runs = 6
    CS = z['csOverTime']
    nTR = np.shape(CS)[1]
    for d in np.arange(nDays):
        day_x = []
        day_dx = []
        if d == 0:
            categSep = CS[0:d1_runs,:,0]
            nRuns = d1_runs
        elif d == 1:
            categSep = CS[0:d2_runs,:,1]
            nRuns = d2_runs
        elif d == 2:
            categSep = CS[0:d3_runs,:,2]
            nRuns = d3_runs
        vec = categSep.flatten()
        day_x.append(vec[0:-1])
        vec_diff = np.diff(vec)
        #a = np.array([0])
        #day_dx.append(np.concatenate([a,vec_diff]))
        day_dx.append(vec_diff)
        day_key = 'day' + str(d)
        all_x[subject_key][day_key] = day_x
        all_dx[subject_key][day_key] = day_dx
        for run in np.arange(nRuns):
            neg_pts = np.where(categSep[run,:]<-.1)[0] # if consecutive negative
            pos_pts = np.where(categSep[run,:]>.1)[0]
            consec_negative = consecutive(neg_pts,stepsize=1)
            neg_area = []
            for g in np.arange(len(consec_negative)):
                    if len(consec_negative[g]) > 1:
                        y_pts = categSep[run,consec_negative[g]]
                        neg_area.append(np.trapz(y_pts,dx=1))
            if len(neg_area)>0:
                neg_area_all[s,run,d] = np.mean(neg_area)
            for n in np.arange(len(neg_pts)):
                this_neg_index = neg_pts[n]
                this_neg_value = categSep[run,this_neg_index]
                # check if index is last one -- add to break into separate blocks within run
                if this_neg_index < len(categSep[run,:])-1:
                    if categSep[run,this_neg_index+1] > this_neg_value:
                        if np.isnan(neg_inc[s,run,d]):
                            neg_inc[s,run,d] = 1
                        else:
                            neg_inc[s,run,d] += 1
                    elif categSep[run,this_neg_index+1] < this_neg_value:
                        if np.isnan(neg_dec[s,run,d]):
                            neg_dec[s,run,d] = 1
                        else:
                            neg_dec[s,run,d] += 1
            for n in np.arange(len(pos_pts)):
                this_pos_index = pos_pts[n]
                this_pos_value = categSep[run,this_pos_index]
                if this_pos_index < len(categSep[run,:])-1:
                    if categSep[run,this_pos_index+1] > this_pos_value:
                        if np.isnan(pos_inc[s,run,d]):
                            pos_inc[s,run,d] = 1
                        else:
                            pos_inc[s,run,d] += 1
                    elif categSep[run,this_pos_index+1] < this_pos_value:
                        if np.isnan(pos_dec[s,run,d]):
                            pos_dec[s,run,d] = 1
                        else:
                            pos_dec[s,run,d] += 1
            # now normalize the counts by total number of negative/positive points in the run
            neg_inc[s,run,d] = neg_inc[s,run,d]/len(neg_pts)
            pos_inc[s,run,d] = pos_inc[s,run,d]/len(pos_pts)
            neg_dec[s,run,d] = neg_dec[s,run,d]/len(neg_pts)
            pos_dec[s,run,d] = pos_dec[s,run,d]/len(pos_pts)

# newest - get most likely denisty for each subject
# position will span -1 to 1 and velocity can span -2 to 2
most_likely_counts = {}
step_size=0.2
pos_edges = np.arange(-1,1+step_size,step_size)
vel_edges = np.arange(-2,2+step_size*2,step_size*2)
all_hist = np.zeros((len(pos_edges)-1,len(vel_edges)-1,nsubs,nDays))
for s in np.arange(nsubs):
    subject_key = 'subject' + str(subjects[s])
    most_likely_counts[subject_key] = {}
    for d in np.arange(nDays):
        day_key = 'day' + str(d)
        pos = all_x[subject_key][day_key][0]
        vel = all_dx[subject_key][day_key][0]
        H, xedges, yedges = np.histogram2d(pos, vel, bins=(pos_edges, vel_edges),density=False)
        H_norm = H/len(pos)
        all_hist[:,:,s,d] = H_norm
        #plt.imshow(H)
        #plt.show()
        most_counts = np.unravel_index(np.argmax(H),np.shape(H))
        most_likely_counts[subject_key][day_key] = most_counts

# now plot each
# day 1
fig,ax = plt.subplots()
plt.subplot(3,2,1)
this_plot = np.nanmean(all_hist[:,:,HC_ind,0],axis=2)
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.imshow(this_plot,cmap='Blues')
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.subplot(3,2,2)
this_plot = np.nanmean(all_hist[:,:,MDD_ind,0],axis=2)
plt.imshow(this_plot,cmap='Blues')
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.subplot(3,2,3)
this_plot = np.nanmean(all_hist[:,:,HC_ind,1],axis=2)
plt.imshow(this_plot,cmap='Blues')
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.subplot(3,2,4)
this_plot = np.nanmean(all_hist[:,:,MDD_ind,1],axis=2)
plt.imshow(this_plot,cmap='Blues')
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.subplot(3,2,5)
this_plot = np.nanmean(all_hist[:,:,HC_ind,2],axis=2)
plt.imshow(this_plot,cmap='Blues')
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.subplot(3,2,6)
this_plot = np.nanmean(all_hist[:,:,MDD_ind,2],axis=2)
plt.imshow(this_plot,cmap='Blues')
CM = scipy.ndimage.measurements.center_of_mass(this_plot)
plt.plot(CM[1],CM[0],'*',color='r', ms=20)
plt.colorbar()

plt.show()
labels_vel = vel_edges.astype(np.float)
labels_vel= np.around(labels_vel,decimals=3).astype(np.str)
labels_pos = pos_edges.astype(np.float)
labels_pos = np.around(labels_pos,decimals=3).astype(np.str)
vmin=-0.015
vmax=0.015
#fig,ax = plt.subplots()
#plt.subplot(1,3,1)
plt.figure(figsize=(20,20))
d=0
this_plot_hc = np.nanmean(all_hist[:,:,HC_ind,d],axis=2).T
this_plot_mdd = np.nanmean(all_hist[:,:,MDD_ind,d],axis=2).T
plt.imshow(this_plot_mdd-this_plot_hc,cmap='bwr',vmin=vmin,vmax=vmax)
plt.yticks(np.arange(len(vel_edges))-0.5,labels_vel,fontsize=10)
plt.xticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.ylabel('d(CS)/dt')
plt.xlabel('CS: scene - face evidence',fontsize=15)
plt.title('Day %i' % d)
#plt.axis('equal')
plt.colorbar()

plt.show()
#plt.subplot(1,3,2)
plt.figure(figsize=(20,10))
d=1
this_plot_hc = np.nanmean(all_hist[:,:,HC_ind,d],axis=2).T
this_plot_mdd = np.nanmean(all_hist[:,:,MDD_ind,d],axis=2).T
plt.imshow(this_plot_mdd - this_plot_hc,cmap='bwr',vmin=vmin,vmax=vmax)
plt.yticks(np.arange(len(vel_edges))-0.5,labels_vel,fontsize=10)
plt.xticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.ylabel('d(CS)/dt')
plt.xlabel('CS: scene - face evidence',fontsize=15)
plt.title('Day %i' % d)
plt.colorbar()
plt.show()
#plt.subplot(1,3,3)
plt.figure(figsize=(20,10))
d=2
this_plot_hc = np.nanmean(all_hist[:,:,HC_ind,d],axis=2).T
this_plot_mdd = np.nanmean(all_hist[:,:,MDD_ind,d],axis=2).T
plt.imshow(this_plot_mdd-this_plot_hc,cmap='bwr',vmin=vmin,vmax=vmax)
plt.yticks(np.arange(len(vel_edges))-0.5,labels_vel,fontsize=10)
plt.xticks(np.arange(len(pos_edges))-0.5,labels_pos,fontsize=10)
plt.ylabel('d(CS)/dt')
plt.xlabel('CS: scene - face evidence',fontsize=15)
plt.title('Day %i' % d)
plt.colorbar()
plt.show()
#plt.show()
# now plot for subject
s = 0
subject_key = 'subject' + str(subjects[s])
fig,ax = plt.subplots()
for d in np.arange(nDays):
    day_key = 'day' + str(d)
    #plt.subplot(1,3,d+1)
#   plt.figure()
    pos = all_x[subject_key][day_key][0]
    vel = all_dx[subject_key][day_key][0]
    #plt.scatter(x,y,alpha=0.5,s = 1.5,color='k')
    sns.jointplot(x=x,y=y,kind='kde',space=0,color='b')
    # pos_range = np.linspace(-1,1,100)
    # vel_range = np.linspace(-2,2,100)
    # posv,velv = np.meshgrid(pos_range,vel_range)
    # pos_deriv = vel[0:-1]
    # vel_deriv = np.diff(vel)
    # plt.quiver(posv,velv,pos_deriv,vel_deriv,alpha=0.75)

    plt.xlim([-1,1])
    plt.ylim([-2,2])
    plt.xlabel('categ sep')
    plt.ylabel('d(categ sep)/dt')
plt.show()

# do 3 separate things for each day
d = 0
mdd_x = []
mdd_dx = []
hc_x = []
hc_dx = []

for s in np.arange(nsubs):
    subject_key = 'subject' + str(subjects[s])
    day_key = 'day' + str(d)
    x = all_x[subject_key][day_key][0]
    y = all_dx[subject_key][day_key][0]
    if subjects[s] < 100:
        hc_x.extend(x)
        hc_dx.extend(y)
    if subjects[s] > 100:
        mdd_x.extend(x.flatten())
        mdd_dx.extend(y)

xnew = hc_x.copy()
xnew.extend(mdd_x)
dxnew = hc_dx.copy()
dxnew.extend(mdd_dx)
data = {}
data['x'] = xnew
data['dx'] = dxnew
subjects = np.ones((len(xnew)))
subjects[0:len(hc_x)] = 0
data['groups'] = subjects
df = pd.DataFrame.from_dict(data)


g = sns.jointplot(x=hc_x,y=hc_dx,kind='kde',space=0,color='k',ratio=3)
#plt.plot([-100,100],[0,0],color='k')
#plt.plot([0,0],[-100,100],color='k')
#g.xlim([-1.5,1.5])
#g.ylim([-1.5,1.5])
plt.show()

g = sns.jointplot(x=mdd_x,y=mdd_dx,kind='kde',space=0,color='r',ratio=3)
#plt.plot([-100,100],[0,0],color='k')
#plt.plot([0,0],[-100,100],color='k')
#g.xlim([-1.5,1.5])
#g.ylim([-1.5,1.5])
plt.show()


i=0
g = sns.JointGrid("x", "dx", df)

for x,group_data in df.groupby("groups"):
    if i==0:
        color = 'k'
        label = 'HC'
    else:
        color = 'r'
        label = 'MDD'
    sns.kdeplot(group_data['x'], ax=g.ax_marg_x, legend=False,color=color)
    sns.kdeplot(group_data['dx'], ax=g.ax_marg_y, vertical=True, legend=False,color=color)
    #g.ax_joint.plot(group_data['x'],group_data['dx'], "o", ms=1,color=color)
    #g.ax_joint.plot(group_data['x'],group_data['dx'], alpha=0.5,color=color)
    #g.ax_joint.scatter(group_data['x'],group_data['dx'],alpha=0.5,color=color)
    #g.ax_joint.kdeplot(group_data['x'],group_data['dx'],alpha=0.5,color=color)
    #g.plot_joint(sns.kdeplot)
    g.plot_joint(sns.distplot,x=group_data['x'],y=group_data['dx'])
    i += 1
plt.show()


# CHANGE COLORMAP
hc = df[df.groups==0]
mdd = df[df.groups==1]
ax = sns.kdeplot(hc.x, hc.dx,cmap="Greys", shade=False, shade_lowest=False,alpha=1,gridsize=100)
ax = sns.kdeplot(mdd.x, mdd.dx, cmap="Greens", shade=False, shade_lowest=False,alpha=1,gridsize=100)
# ax = sns.jointplot(hc.x, hc.dx, kind="hex", cmap="Greys")
# ax = sns.jointplot(mdd.x, mdd.dx, kind="hex", color="#4CB391")
ax.set_xlim([-1.4,1.4])
ax.set_ylim([-0.35,0.35])
ax.set_xlabel('category separation')
ax.set_ylabel('d(x)/dt')
plt.show()


g = sns.JointGrid("x", "dx", df)
sns.kdeplot(x='x',y='dx', ax=g.ax_marg_x, legend=False,data=df)
sns.kdeplot(day_tips["tip"], ax=g.ax_marg_y, vertical=True, legend=False)
    g.ax_joint.plot(day_tips["total_bill"], day_tips["tip"], "o", ms=5)


tips = sns.load_dataset("tips")
g = sns.JointGrid("total_bill", "tip", tips)
for x, group_data in df.groupby("groups"):
    sns.kdeplot(group_data['x'], ax=g.ax_marg_x, legend=False)
    sns.kdeplot(group_data['dx'], ax=g.ax_marg_y, vertical=True, legend=False)
    g.ax_joint.plot(group_data['x'],group_data['dx'], "o", ms=1)
plt.show()
tips = sns.load_dataset("tips")

for d in np.arange(nDays):
    fig,ax = plt.subplots()
    for s in np.arange(nsubs):
        subject_key = 'subject' + str(subjects[s])
        day_key = 'day' + str(d)
        if subjects[s] < 100:
            plt.subplot(1,2,1)
        elif subjects[s] > 100:
            plt.subplot(1,2,2)
        x = all_x[subject_key][day_key][0]
        y = all_dx[subject_key][day_key][0]
        #plt.scatter(x,y,alpha=0.5,s = 1.5,color='k')
        plt.plot(x,y,lw=1,alpha=0.1,color='k')
        plt.plot([-100,100],[0,0],color='k')
        plt.plot([0,0],[-100,100],color='k')
        plt.xlim([-1.5,1.5])
        plt.ylim([-2,2])
        plt.xlabel('categ sep')
        plt.ylabel('d(categ sep)/dt')
    plt.show()

fig,ax = plt.subplots()
for d in np.arange(nDays):
    plt.subplot(1,3,d+1)
    for s in np.arange(nsubs):
        subject_key = 'subject' + str(subjects[s])
        day_key = 'day' + str(d)
        if subjects[s] < 100:
            color='k'
        elif subjects[s] > 100:
            color='r'
        x = all_x[subject_key][day_key][0]
        y = all_dx[subject_key][day_key][0]
        plt.scatter(x,y,alpha=0.1,s = 1,color=color)
        plt.plot([-100,100],[0,0],color='k')
        plt.plot([0,0],[-100,100],color='k')
        plt.xlim([-1.5,1.5])
        plt.ylim([-2,2])
        plt.xlabel('categ sep')
        plt.ylabel('d(categ sep)/dt')
plt.show()

plt.figure()
plt.plot(x,label='x')
plt.plot(y,  label='dx')
plt.legend()
plt.show()

plt.figure()
#plt.plot(x,y, '-o', lw=10)
plt.scatter(x,y,alpha=0.5,s = 1.5)
plt.plot([-100,100],[0,0],color='k')
plt.plot([0,0],[-100,100],color='k')
plt.xlim([-1.5,1.5])
plt.ylim([-2,2])
plt.xlabel('categ sep')
plt.ylabel('d(categ sep)/dt')
plt.show()
# everything is 0 for some reason :()
# plot by run 
s=0
d=0
alpha=0.5
fig,ax = plt.subplots(1,3,sharey='col')
d=0
for s in np.arange(nsubs):
    if s in HC_ind:
        color = 'k'
    elif s in MDD_ind:
        color = 'r'
    ax[0].plot(neg_dec[s,:,d],color=color,alpha=alpha)
    ax[0].set_xlim([0,d1_runs-1])
hc_mean = np.nanmean(neg_dec[HC_ind,0:d1_runs,d],axis=0)
mdd_mean = np.nanmean(neg_dec[MDD_ind,0:d1_runs,d],axis=0)
ax[0].plot(hc_mean,color='k',lw=5,label='MDD')
ax[0].plot(mdd_mean,color='r',lw=5,label='MDD')

d=1
for s in np.arange(nsubs):
    if s in HC_ind:
        color = 'k'
    elif s in MDD_ind:
        color = 'r'
    ax[1].plot(neg_dec[s,:,d],color=color,alpha=alpha)
    ax[1].set_xlim([0,d2_runs-1])
hc_mean = np.nanmean(neg_dec[HC_ind,0:d2_runs,d],axis=0)
mdd_mean = np.nanmean(neg_dec[MDD_ind,0:d2_runs,d],axis=0)
ax[1].plot(hc_mean,color='k',lw=5,label='MDD')
ax[1].plot(mdd_mean,color='r',lw=5,label='MDD')

d=2
for s in np.arange(nsubs):
    if s in HC_ind:
        color = 'k'
    elif s in MDD_ind:
        color = 'r'
    ax[2].plot(neg_dec[s,:,d],color=color,alpha=alpha)
    ax[2].set_xlim([0,d3_runs-1])
hc_mean = np.nanmean(neg_dec[HC_ind,0:d3_runs,d],axis=0)
mdd_mean = np.nanmean(neg_dec[MDD_ind,0:d3_runs,d],axis=0)
ax[2].plot(hc_mean,color='k',lw=5,label='MDD')
ax[2].plot(mdd_mean,color='r',lw=5,label='MDD')

plt.show()

# average over day
neg_dec_day = np.nanmean(neg_dec,axis=1)
stat = neg_dec_day
fig = plotPosterStyle(stat,subjects)
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.01)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(stat),0.01)
x,y = nonNan(stat[HC_ind,2],stat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,2,np.nanmax(stat),0.01)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat(p/2,0,2,np.nanmax(stat),0.03)
plt.ylabel('p(decrease|negative)')
plt.xticks(np.arange(3))
plt.show()

mdd_mean = np.mean(neg_dec_day[MDD_ind,:],axis=0)
hc_mean = np.mean(neg_dec_day[HC_ind,:],axis=0)
mdd_err = scipy.stats.sem(neg_dec_day[MDD_ind,:],axis=0)
hc_err = scipy.stats.sem(neg_dec_day[HC_ind,:],axis=0)
alpha=0.3
plt.figure()
for s in np.arange(nsubs):
    if s in HC_ind:
        color = 'k'
    elif s in MDD_ind:
        color = 'r'
    plt.plot(neg_dec_day[s,:],'-',ms=10,color=color,alpha=alpha,lw=2)
plt.errorbar(x=np.arange(nDays),y=hc_mean,yerr=hc_err,color='k',lw=2,label='MDD',fmt='-o',ms=10)
plt.errorbar(x=np.arange(nDays),y=mdd_mean,yerr=mdd_err,color='r',lw=2,label='MDD',fmt='-o',ms=10)
plt.xlabel('day')
plt.ylabel('p(decrease| negative)')
plt.xticks([0,1,2])
plt.show()

# difference between day 1 and day 3?
scipy.stats.ttest_rel(neg_dec_day[MDD_ind,0],neg_dec_day[MDD_ind,2])
scipy.stats.ttest_rel(neg_dec_day[HC_ind,0],neg_dec_day[HC_ind,2])

# does this change from day 3 to day 1 predict MADRS at the end?

neg_dec_day = np.nanmean(neg_area_all,axis=1)
stat = neg_dec_day
fig = plotPosterStyle(stat,subjects)
x,y = nonNan(stat[HC_ind,0],stat[MDD_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,0,np.nanmax(stat),0.03)
x,y = nonNan(stat[HC_ind,1],stat[MDD_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,1,np.nanmax(stat),0.03)
x,y = nonNan(stat[HC_ind,2],stat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addSingleStat(p/2,2,np.nanmax(stat),0.03)
x,y = nonNan(stat[MDD_ind,0],stat[MDD_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat(p/2,0,2,np.nanmax(stat),1)

plt.ylabel('Area under -0.1')
plt.xticks(np.arange(3))
plt.show()



mdd_mean = np.mean(neg_dec_day[MDD_ind,:],axis=0)
hc_mean = np.mean(neg_dec_day[HC_ind,:],axis=0)
mdd_err = scipy.stats.sem(neg_dec_day[MDD_ind,:],axis=0)
hc_err = scipy.stats.sem(neg_dec_day[HC_ind,:],axis=0)
alpha=0.3
plt.figure()
for s in np.arange(nsubs):
    if s in HC_ind:
        color = 'k'
    elif s in MDD_ind:
        color = 'r'
    plt.plot(neg_dec_day[s,:],'-',ms=10,color=color,alpha=alpha,lw=2)
plt.errorbar(x=np.arange(nDays),y=hc_mean,yerr=hc_err,color='k',lw=2,label='MDD',fmt='-o',ms=10)
plt.errorbar(x=np.arange(nDays),y=mdd_mean,yerr=mdd_err,color='r',lw=2,label='MDD',fmt='-o',ms=10)
plt.xlabel('day')
plt.ylabel('area under -0.1')
plt.xticks([0,1,2])
plt.show()

# difference between day 1 and day 3?
scipy.stats.ttest_rel(neg_dec_day[MDD_ind,0],neg_dec_day[MDD_ind,2])
scipy.stats.ttest_rel(neg_dec_day[HC_ind,0],neg_dec_day[HC_ind,2])

colors=['k','r']
M = getMADRSscoresALL()
fig = plt.figure(figsize=(10,7))
for s in np.arange(nsubs):
    subjectNum  = subjects[s]
    this_sub_madrs = M[subjectNum]
    madrs_change = this_sub_madrs[1] - this_sub_madrs[0]
    if subjectNum < 100:
        style = 0
    elif subjectNum > 100:
        style = 1
    plt.plot(neg_dec_day[s,2] - neg_dec_day[s,0],madrs_change,marker='.',ms=20,color=colors[style],alpha=0.5)
plt.xlabel('p(decrease|negative) Change 3 - 1')
plt.ylabel('MADRS Change 3 - 1')
plt.show()
#scipy.stats.pearsonr(neg_dec_day[MDD_ind,2] - neg_dec_day[MDD_ind,0])
