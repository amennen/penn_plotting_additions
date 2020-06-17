# common functions
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
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
font = {'size': 22,
        'weight': 'normal'}
plt.rc('axes', linewidth=3)
plt.rc('xtick.major', size=10, width = 2)
plt.rc('ytick.major', size=10, width = 2)
# plt.rc('xtick.major', size=0, width = 3)
# plt.rc('ytick.major', size=0, width = 3)
# import pandas as pd
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
import matplotlib.pyplot as plt
import math


def printStatsResults(text,r,p,x,y):
  print('*******************')
  print('group size x: %i; group size y: %i' % (len(x),len(y)))
  text_stat = "{0}: t = {1:03f}, p = {2:03f}".format(text,r,p)
  print(text_stat)
  return 


def convertMatToDF(all_data,all_subjects):
    data = {}
    data_vector = all_data.flatten() # now it's in subj day 1, subj day 2, etc.
    n_days = np.shape(all_data)[1]
    n_subs = len(all_subjects)
    subject_vector = np.repeat(all_subjects,n_days)
    day_vector = np.tile(np.arange(n_days),n_subs)
    group_vector1 = np.zeros((n_subs,))
    MDD_ind = np.argwhere(all_subjects>100)[:,0]
    group_vector1[MDD_ind] = 1
    group_vector = np.repeat(group_vector1,n_days)
    data['data'] = data_vector
    data['subjects'] = subject_vector
    data['group'] = group_vector
    data['day'] = day_vector
    df = pd.DataFrame.from_dict(data)
    return df

def convertMatToDF_valence(all_data,subjects):
    # let's assume all_data is in the shape ((nsubjects,n_days,nemotions))
    data = {}
    data_vector = all_data.flatten() # now it's in subj day 1, subj day 2, etc.
    n_days = np.shape(all_data)[1]
    n_subs = len(subjects)
    nemotions = np.shape(all_data)[2]
    subject_vector = np.repeat(subjects,n_days*nemotions)
    day_vector = np.tile(np.repeat(np.arange(n_days),nemotions),n_subs)
    day_vector_string = ['Pre NF'] * len(day_vector)
    days_string = ['Pre NF', 'Mid NF', 'Post NF', '1M FU']
    for d in np.arange(len(day_vector)):
      if day_vector[d] > 0:
        if day_vector[d] == 1:
          if n_days == 3:
            day_vector_string[d] = days_string[2]
          elif n_days == 4:
            day_vector_string[d] = days_string[1]
        elif day_vector[d] == 2:
          if n_days == 3:
            day_vector_string[d] = days_string[3]
          else:
            day_vector_string[d] = days_string[2]
        elif day_vector[d] == 3:
          day_vector_string[d] = days_string[3]
    group_vector1 = np.zeros((n_subs,))
    MDD_ind = np.argwhere(subjects>100)[:,0]
    group_vector1[MDD_ind] = 1
    group_vector = np.repeat(group_vector1,n_days*nemotions)
    MDD_ind = np.argwhere(group_vector==1)[:,0].astype(int)
    group_string = ['HC'] * len(group_vector)
    for i in np.arange(len(group_vector)):
      if i in MDD_ind:
        group_string[i]='MDD'
    emotions_vector = np.tile(np.arange(nemotions),n_subs*n_days)
    data['data'] = data_vector
    data['subjects'] = subject_vector
    data['group'] = group_string
    data['day'] = day_vector_string
    data['emotion'] = emotions_vector
    df = pd.DataFrame.from_dict(data)
    return df

def makeColorPalette(colors):
  # Create an array with the colors you want to use
  # Set your custom color palette
  customPalette = sns.color_palette(colors)
  return customPalette

def plotPosterStyle_DF(all_data,subjects):
  df = convertMatToDF(all_data,subjects)
  fig,ax = plt.subplots(figsize=(12,9))
  sns.despine()
  P1 = makeColorPalette(['#636363','#de2d26'])
  P2 = makeColorPalette(['#f0f0f0','#fee0d2'])
  P3 = makeColorPalette(['#bdbdbd','#fc9272'])
  #sns.set_palette(sns.color_palette(colors))
  sns.barplot(data=df,x='day',y='data',hue='group',ci=68,linewidth=2.5,palette=P1)#errcolor=".2", edgecolor=".2")
  #sns.barplot(data=df,x='day',y='data',hue='group',ci=68,linewidth=2.5,palette=P1,errcolor=".2", edgecolor=".2")
  sns.swarmplot(data=df,x='day',y='data',hue='group',split=True,palette=P3,size=8)
  ax.get_legend().remove()
  #plt.show()
  return fig,ax

def plotPosterStyle_DF_valence(all_data,subjects,emo,ylabel):
  df = convertMatToDF_valence(all_data,subjects)
  df = df.rename(columns={'data':ylabel})
  fig,ax = plt.subplots(figsize=(12,9))
  sns.despine()
  #P1 = makeColorPalette(['#636363','#de2d26'])
  #P2 = makeColorPalette(['#f0f0f0','#fee0d2'])
  #P3 = makeColorPalette(['#bdbdbd','#fc9272'])
  #sns.set_palette(sns.color_palette(colors))
  g=sns.catplot(data=df,x="day",y=ylabel,hue="emotion",col="group",ci=68,kind='bar',palette="Blues_d")
  for t, l in zip(g._legend.texts, emo): t.set_text(l)
  # plt.show()

  # plt.subplot(1,2,1)
  # g=sns.barplot(data=df[df['group']==0],x='day',y='data',hue='emotion',ci=68,linewidth=2.5, palette="Blues_d")#errcolor=".2", edgecolor=".2")
  # for t, l in zip(g._legend.texts, emo): t.set_text(l)
  # #sns.barplot(data=df,x='day',y='data',hue='group',ci=68,linewidth=2.5,palette=P1,errcolor=".2", edgecolor=".2")
  # sns.swarmplot(data=df[df['group']==0],x='day',y='data',hue='emotion',split=True,size=8, palette="Blues_d")
  # #ax.legend().remove()
  # plt.subplot(1,2,2)
  # sns.barplot(data=df[df['group']==1],x='day',y='data',hue='emotion',ci=68,linewidth=2.5, palette="Blues_d")#errcolor=".2", edgecolor=".2")
  # #sns.barplot(data=df,x='day',y='data',hue='group',ci=68,linewidth=2.5,palette=P1,errcolor=".2", edgecolor=".2")
  # sns.swarmplot(data=df[df['group']==1],x='day',y='data',hue='emotion',split=True,size=8, palette="Blues_d")
  # #ax.legend().remove()
  # plt.show()
  return fig,ax

def plotPosterStyle_DF_valence_separateGroups(data1,data2,subjects,group,emo,ylabel,labels):
  #combine data first - data1 is 25 subjects x 3 days x 4 emotions
  df1 = convertMatToDF_valence(data1,subjects)
  df2 = convertMatToDF_valence(data2,subjects)
  df1['measure'] = [labels[0]] * len(df1)
  df2['measure'] = [labels[1]] * len(df2)
  combined_df = df1.append(df2)
  df = combined_df.rename(columns={'data':ylabel})
  fig,ax = plt.subplots(figsize=(12,9))
  #sns.despine()
  #plt.subplot(2,1,1)
  g=sns.catplot(data=df[df['group']==group],x="emotion",y=ylabel,hue="measure",col="day",ci=68,kind='bar',palette="Blues_d")
  #plt.title('HC')
  #plt.xticks
  #plt.show()
  #plt.subplot(2,1,2)
  # fig,ax = plt.subplots(figsize=(12,9))
  # g=sns.catplot(data=df[df['group']=='MDD'],x="emotion",y=ylabel,hue="measure",col="day",ci=68,kind='bar',palette="Blues_d")
  # #for t, l in zip(g._legend.texts, emo): t.set_text(l)
  #plt.title('MDD')
  #plt.show()
  return fig,ax

def plotPosterStyle(all_data,subjects):
    """Assume data is in subject x day """
    HC_ind = np.argwhere(subjects<100)[:,0]
    MDD_ind = np.argwhere(subjects>100)[:,0]
    HC_data = all_data[HC_ind,:]
    MDD_data = all_data[MDD_ind,:]
    n_subs = len(subjects)
    hc_mean = np.nanmean(HC_data,axis=0)
    mdd_mean = np.nanmean(MDD_data,axis=0)
    hc_err = scipy.stats.sem(HC_data,axis=0,nan_policy='omit')
    mdd_err = scipy.stats.sem(MDD_data,axis=0,nan_policy='omit')
    alpha=0.2
    n_days = np.shape(all_data)[1]
    fig,ax = plt.subplots()
    sns.despine()
    for s in np.arange(n_subs):
        if s in HC_ind:
            color = 'k'
        elif s in MDD_ind:
            color = 'r'
        plt.plot(all_data[s,:],'-',ms=10,color=color,alpha=alpha,lw=2)
    plt.errorbar(x=np.arange(n_days),y=hc_mean,yerr=hc_err,color='k',lw=5,label='HC',fmt='-o',ms=10)
    plt.errorbar(x=np.arange(n_days),y=mdd_mean,yerr=mdd_err,color='r',lw=5,label='MDD',fmt='-o',ms=10)
    plt.xlabel('day')
    #plt.ylabel('area under -0.1')
    plt.xticks(np.arange(n_days))
    plt.legend()
    #plt.show()
    return fig

def plotPosterStyle_multiplePTS(all_data,subjects,plotAll=1):
    """Assume data is in subject x PTS x day """
    colors_dark = ['#636363','#de2d26']
    colors_light = ['#636363','#de2d26']
    HC_ind = np.argwhere(subjects<100)[:,0]
    MDD_ind = np.argwhere(subjects>100)[:,0]
    HC_data = all_data[HC_ind,:,:]
    MDD_data = all_data[MDD_ind,:,:]
    n_subs = len(subjects)
    hc_mean = np.nanmean(HC_data,axis=0)
    mdd_mean = np.nanmean(MDD_data,axis=0)
    hc_err = scipy.stats.sem(HC_data,axis=0,nan_policy='omit')
    mdd_err = scipy.stats.sem(MDD_data,axis=0,nan_policy='omit')
    alpha=0.2
    n_days = np.shape(all_data)[2]
    nPoints = np.shape(all_data)[1]
    fig,ax = plt.subplots(figsize=(17,9))
    all_axes = []
    for d in np.arange(n_days):
      ax_n = plt.subplot(1,n_days,d+1)
      sns.despine()
      if plotAll:
        for s in np.arange(n_subs):
            if s in HC_ind:
                color = 0
            elif s in MDD_ind:
                color = 1
            plt.plot(all_data[s,:,d],'-',ms=10,color=colors_light[color],alpha=alpha,lw=2)
      plt.errorbar(x=np.arange(nPoints),y=hc_mean[:,d],yerr=hc_err[:,d],color=colors_dark[0],lw=5,label='HC',fmt='-o',ms=10)
      plt.errorbar(x=np.arange(nPoints),y=mdd_mean[:,d],yerr=mdd_err[:,d],color=colors_dark[1],lw=5,label='MDD',fmt='-o',ms=10)
      plt.xlabel('point')
      #plt.ylabel('area under -0.1')
      plt.xticks(np.arange(nPoints))
      all_axes.append(ax_n)
    return fig, all_axes

def addComparisonStat(score,x1,x2,maxHeight,heightAbove):
  y,h,col = maxHeight + heightAbove, heightAbove, 'k'
  plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
  text_stat = "1-sided p = {0:2.4f}".format(score)
  plt.text((x1+x2)*.5, y+h, text_stat, ha='center', va='bottom', color=col,fontsize=25)
  return

def addComparisonStat_SYM(score,x1,x2,maxHeight,heightAbove,fontH,text_above=[],addlines=1):
  y,h,col = maxHeight, heightAbove, 'k'
  y2 = y+h
  h2 = y+h+h
  if score < 0.0001:
    text = '****'
  elif score < 0.001:
    text = '***'
  elif score < 0.01:
    text = '**'
  elif score < 0.05:
    text = '*'
  elif score < 0.1:
    text = '+'
  elif score > 0.1:
    text = 'ns'
  if score < 0.1: # don't write n.s.
    if x1 != x2: # only add the lines if spanning x spots
      if addlines:
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
      else:
        plt.plot([x1, x1, x2, x2], [y, y, y, y], lw=1.5, c=col)
    if len(text_above) > 0:
      text_stat = "{}\n{}".format(text_above,text)
    else:
      text_stat = text
    plt.text((x1+x2)*.5, y+h+(fontH), text_stat, ha='center', va='bottom', color=col,fontsize=22,fontstyle='normal')
  return

def addSingleStat(score,x,maxHeight,heightAbove):
  y,h,col = maxHeight + heightAbove, heightAbove, 'k'
  plt.plot([x, x], [y, y+h], lw=1.5, c=col)
  text_stat = "1-sided p = {0:2.4f}".format(score)
  plt.text(x, y+h, text_stat, ha='center', va='bottom', color=col,fontsize=15)
  return

def addSingleStat_ax(score,x,maxHeight,heightAbove,ax):
  y,h,col = maxHeight + heightAbove, heightAbove, 'k'
  ax.plot([x, x], [y, y+h], lw=1.5, c=col)
  text_stat = "p = {0:2.4f}".format(score)
  ax.text(x, y+h, text_stat, ha='center', va='bottom', color=col,fontsize=15)
  return

def nonNan(x,y):
  if len(x) == len(y): # then treat as paired case
    if any(np.isnan(x)) and not any(np.isnan(y)):
        xnew = x[~np.isnan(x)]
        ynew = y[~np.isnan(x)]
    elif any(np.isnan(y)) and not any(np.isnan(x)):
        ynew = y[~np.isnan(y)]
        xnew = x[~np.isnan(y)]
    elif any(np.isnan(x)) and any(np.isnan(y)):
        bad_x = np.argwhere(np.isnan(x))
        bad_y = np.argwhere(np.isnan(y))
        all_bad = np.unique(np.concatenate((bad_x,bad_y)))
        all_indices = np.arange(len(x))
        keep_indices = np.delete(all_indices,all_bad)
        xnew = x[keep_indices]
        ynew=y[keep_indices]
    elif not any(np.isnan(x)) and not any(np.isnan(y)):
        print('NO NANS!')
        xnew=x
        ynew=y
  else:
    # if they are different numbers just remove what is nan
    if any(np.isnan(x)):
      xnew = x[~np.isnan(x)]
    else:
      xnew = x
    if any(np.isnan(y)):
      ynew = y[~np.isnan(y)]
    else:
      ynew = y
  return xnew,ynew
