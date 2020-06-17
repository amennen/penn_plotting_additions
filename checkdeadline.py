# purpose: check if would have made the deadline
import numpy as np
import os, re
#import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy import io
import pandas as pd
import glob
projectdir = '/home/amennen/code/rtAttenPenn/data/'
deltat = 0.1 # how long to wait after starts looking
subject = 106
d=2
day=d+1
subjectdir = projectdir + '/subject' + str(subject) + '/day' + str(day)
runs = [7,9,8]
ndays=3
for r in np.arange(1,runs[d]):
	subrun = r+1
	rundir = subjectdir + '/run' + str(subrun) 
	os.chdir(rundir)
	os.chdir(rundir + '/classoutput')
	os.system("find . -name 'vol*' -printf '%p %C@\n' >> matpytiming_epoch.txt")
	os.chdir(rundir)
	fn = glob.glob('blockdata_*.mat')
	z = scipy.io.loadmat(fn[0])
	# this is in shape block x trial #
	classifierLoadStart = z['timing']['classifierLoadStart'][0][0]
	oddTrials = np.arange(3,50,2)-1
	files = z['blockData']['classOutputFile']
	fileprocessname = glob.glob('patternsdata_*.mat')
	z = scipy.io.loadmat(fileprocessname[0])
	timing_foundfiles = z['timing']['foundDicom'][0][0]
	names_foundfiles = z['patterns']['newFile'][0][0]
	# make dict of when created TR
	TR_creation = {}
	RT_ind = np.argwhere(np.isnan(timing_foundfiles[0,:])==0)
	for t in np.arange(np.shape(RT_ind)[0]):
		thisIndex = RT_ind[t][0]
		this_name = names_foundfiles[0,thisIndex][0]
		file_name = this_name[-7:-4]
		volstr = 'vol_' + this_name[-7:-4]
		TR_creation[volstr] = timing_foundfiles[0,thisIndex] 

	# now get timing data from previous code--timingcheck
	file = open('classoutput/matpytiming_epoch.txt')
	mat_timing = {}
	py_timing = {}
	for line in file:
		if 'vol' in line:
			print(line)
			temp_parts = str.split(line, ' ')
			vol_name = temp_parts[0][2:9]
			fulltime = temp_parts[1][0:-1]
			if 'mat' in line:
				mat_timing[vol_name] = fulltime
			elif 'py' in line:
				py_timing[vol_name] = fulltime

			# now we're just looking at the matlab and python load time


	# now go through each volume and get the time
	for iBlock in np.arange(4,8):
		startTimes = classifierLoadStart[iBlock,:]
		filenames = files[0,iBlock][0,1:]
		print('**************')
		for t in np.arange(len(oddTrials)):

			index = oddTrials[t]
			deadline = startTimes[index] + deltat
			thisfile = filenames[t][0][0:-4]
			mat_time = mat_timing[thisfile]
			py_time = py_timing[thisfile]
			found_time = TR_creation[thisfile]
			wait_mat = deadline - float(mat_time)
			wait_py = deadline - float(py_time)
			class_time = float(py_time) - found_time
			#print(wait_mat)
			print(wait_py)
			if wait_py < 0:
				print('WOULD HAVE MISSED!!')
			# now print classify - found time
			#print('classify - found = %4.4f' %(class_time))