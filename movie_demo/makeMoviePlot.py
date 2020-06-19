##### CODE TO MAKE DEMO MOVIE #####
# conda install ffmpeg x264=20131218 -c conda-forge
# get data from:
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import sys
import glob
import matplotlib.pylab as pl
import os
sys.path.append(os.getcwd())
import rtfMRI.utils as utils
from rtfMRI.utils import loadMatFile
from rtfMRI.RtfMRIClient import loadConfigFile
font = {'weight':'normal',
'size':22}
plt.rc('xtick.major', size=10, width = 5)
plt.rc('ytick.major', size=10, width = 5)
plt.rc('axes',linewidth=5)
plt.rc('font', **font)
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation
from celluloid import Camera
def getBlockData(path, run, block):
	# run and block are in matlab indices
	filename = "{0}/blockdata_{1}_*.mat".format(path,run)
	fn = glob.glob(filename)[-1]
	data = loadMatFile(fn)
	smoothAttImgProp = data['blockData']['smoothAttImgProp'] 
	this_block_smoothAttImgProp = smoothAttImgProp[0,block-1].flatten('F')
	categSep = data['blockData']['categsep']  
	this_block_categSep = categSep[0,block-1][0]
	return this_block_categSep, this_block_smoothAttImgProp

def getPatternsData(path, run, block):
	# run and block are in matlab indices
	if block <= 4:
		filename = "{0}/blkGroup_r{1}_p1_*.mat".format(path,run)
	else:
		filename = "{0}/blkGroup_r{1}_p2_*.mat".format(path,run)
	fn = glob.glob(filename)[-1]
	data = loadMatFile(fn)
	categSep = data['patterns']['categoryseparation'][0,:]
	categSep_nonNan, [] = nonNan(categSep,[])
	categSep_blocks = categSep_nonNan.reshape(4,25)
	if block > 4:
		blockInd = block - 5
	else:
		blockInd = block - 1
	this_block_categSep = categSep_blocks[blockInd,:]
	return this_block_categSep
	# reshape to be 


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


def updateFig(*args):
	global stim_index, trial_index
	stim_index +=1
	trial_index = stim_index - 2
	return

movie_path = 'anne_additions/movie_demo/'
stimuli_path = movie_path + 'stimuli_cropped'
data_path = movie_path + 'data'
runNum = 7 # matlab index
blockNum = 8 # matlab index
categSep, smoothAttImgProp = getBlockData(data_path, runNum, blockNum)
pattern_categSep = getPatternsData(data_path, runNum, blockNum) # use this from TR 3 on
categSep_nonNan, [] = nonNan(categSep,[])
trial_vec_cs = np.arange(5,51,2) 
categ_sep_use = pattern_categSep[0:-2]
nTrials = 50
trial_vec = np.arange(1,nTrials+1)
xticks = np.arange(10,60,10)
yticks_cs = np.array([-1,0,1])
yticks_p = np.array([0,0.5,1])


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(1, 2, 1, xticklabels=[], yticklabels=[], xticks=[], yticks=[])
ax2 = fig.add_subplot(2, 2, 2, xticks=xticks, yticks=yticks_cs)
ax3 = fig.add_subplot(2, 2, 4, xticks=xticks, yticks=yticks_p)
camera = Camera(fig)
ims = []
for stim_index in np.arange(nTrials+2):
	trial_index = stim_index - 2
	if stim_index == 0:
		fn_stim = "{0}/instruct.png".format(stimuli_path)
	elif stim_index == 1:
		fn_stim = "{0}/fix.png".format(stimuli_path)
	elif stim_index > 1:
		fn_stim = "{0}/s{1}.png".format(stimuli_path,trial_index+1)
	img = mpimg.imread(fn_stim)
	im1 = ax1.imshow(img)
	ax1.set_title('')
	#ax1.set_title('display')

	cmap = ListedColormap(['r','k'])
	norm = BoundaryNorm([-1, 0, 1], cmap.N)
	ax2.spines['right'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax2.set_ylabel('')
	#ax2.set_ylabel('scene - face classification',fontsize=17)
	if trial_index > 0:
		inds_to_plot = np.argwhere(trial_vec_cs<=trial_index+1)[:,0]
		if len(inds_to_plot):#
			ax2.plot(trial_vec_cs[inds_to_plot],categ_sep_use[inds_to_plot], marker='.', markersize=10, linewidth=4, color='k')
	X = [[1, 1], [0, 0]]
	im2 = ax2.imshow(X, interpolation = 'bicubic', cmap=pl.cm.RdGy, extent = (0,50,-1.05, 1.05), alpha=0.5, aspect = "auto")
	ax2.set_xlim([0.5,nTrials+0.5])
	ax2.set_ylim([-1.05,1.05])

	ax3.spines['right'].set_visible(False)
	ax3.spines['top'].set_visible(False)
	ax3.set_xlabel('')
	ax3.set_ylabel('')
	#ax3.set_xlabel('trial')
	#ax3.set_ylabel('scene proportion',fontsize=17)
	ax3.set_xlim([0.5,nTrials+0.5])
	ax3.set_ylim([-.025,1.025])
	if trial_index >= 0:
		ax3.plot(trial_vec[0:trial_index+1],smoothAttImgProp[0:trial_index+1], marker='.', markersize=10, linewidth=4, color = 'k')
	ax3.imshow(X, interpolation = 'bicubic', cmap=pl.cm.RdGy, extent = (0,50, -.025, 1.025), alpha=0.5, aspect = "auto")
	camera.snap()
	#fn = 'movie_screenshots/snap_{0}.eps'.format(stim_index)
	#plt.savefig(fn)
animation = camera.animate()

fn1 = movie_path + 'rtAttenDemo_fps1.mp4'
fn2 = movie_path + 'rtAttenDemo_fps2.mp4'
animation.save(fn1,fps=1)
animation.save(fn2,fps=2)




