# purpose: compare timing with Matlab/python outputs for rtAttention experiment

import numpy as np
import os, re
#import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
subject=106
projectdir = '/home/amennen/code/rtAttenPenn/data/'

runs = [7,9,8]
ndays=3
#subject_times_day = np.zeros(ndays,115)
#os.system('stat -c %N%y * >> matpytiming.txt')
subject_times = np.empty(1,)
for d in np.arange(3):
	day = d + 1
	subjectdir = projectdir + '/subject' + str(subject) + '/day' + str(day)
	totalruns = runs[d] - 1
	for r in np.arange(1,runs[d]):
		subrun = r + 1
		if day == 2 and subrun == 8:
			pass
		else:
			rundir = subjectdir + '/run' + str(subrun) + '/classoutput'
			os.chdir(rundir)
			#os.system('stat -c %N%y * >> matpytiming4.txt')
			file=open('matpytiming4.txt')
			i=0
			allnum = np.zeros((500,))
			tdiff = np.zeros((500,))
			for line in file:
				if 'vol' in line:
					temp = re.findall('\d+', line)
					fulltime = int(temp[4])*3600 + int(temp[5])*60 + int(temp[6]) + int(temp[7])/1E9
					allnum[i] = fulltime
					if i % 2: 
						tdiff[int((i-1)/2)] = allnum[i] - allnum[i-1]
						# python create time - matlab create time
					i+=1

			alltimepoints = tdiff[0:int(i/2)]
			print('*************************')
			print('looking at day %i and run %r' % (day,subrun))
			print(alltimepoints)
			print('*************************')
			subject_times = np.concatenate((subject_times,alltimepoints),axis=0)
		
alltimes = subject_times[1:]
alldays = np.concatenate((np.ones(6*115,), 2*np.ones(7*115,), 3*np.ones(7*115,)),axis=0)
#datamat = np.concatenate((alltimes[:,np.newaxis,alldays,np.newaxis),axis=1)
day1_ind = np.argwhere(alldays==1)
day2_ind = np.argwhere(alldays==2)
day3_ind = np.argwhere(alldays==3)
m = np.mean(subject_times)
plt.figure(figsize=(10,7))
#plt.plot([1,2,3,4], [1,4,9,16], 'ro')
#plt.show()
plt.plot([m,m], [0, 200], 'r--')
plt.title('Subject %i: Day  '% (subject))
plt.hist(subject_times[day1_ind],label='day1',alpha=0.5)
plt.hist(subject_times[day2_ind],label='day2',alpha=0.5)
plt.hist(subject_times[day3_ind],label='day3',alpha=0.5)
plt.xlabel('Python Time - Matlab Time')
plt.legend()
plt.show()

df = pd.DataFrame(data=datamat,columns=['diff', 'day'])
plt.figure()
sns.barplot(data=df, x='day', y='diff')


plt.xlabel('Run Number')
plt.ylabel('XVAL AUC')
plt.ylim([0,1])
#plt.legend(('matlab', 'python'))
plt.show()



df = pd.DataFrame(data=datamat,columns=['acc', 'matpy', 'runnum'])

plt.figure()
sns.barplot(data=df, x='runnum', y='acc', hue='matpy')
plt.title('Subject %i: Day %i '% (subjectNum,dayNum))
plt.xlabel('Run Number')
plt.ylabel('XVAL AUC')
plt.ylim([0,1])
#plt.legend(('matlab', 'python'))
plt.show()

#file=open('fulltime.txt')
#i=0
#allnum = np.zeros((500,))
#tdiff = np.zeros((200,))
#for line in file:
#	if 'vol' in line:
#		allnum[i] = int(re.findall('\d+',line)[1])
#		if i % 2: 
#			tdiff[int((i-1)/2)] = allnum[i] - allnum[i-1]
#		i+=1
	
#alltimepoints = tdiff[0:int(i/2)] % 115 samples per day per subject

