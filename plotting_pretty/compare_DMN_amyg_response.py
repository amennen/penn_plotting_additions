DMN_subjects = np.array([3,4,5,6,7,8,9,10,11,106,107,108,109,110,111,112,113,114])
all_subjects = allsubjects = np.array([1,2,3,4,5,6,7,8,9,10,11,101,102,103,104,105,106,107,108,109,110,111,112,113,114])
colors = ['#636363','#de2d26']

# DMN day 1 and AMY Day 1
#dmn_connectivity all_subject_averages_fearful
amyg = all_subject_averages_fearful[:,1,:] - all_subject_averages_fearful[:,0,:]
dmn_change = dmn_connectivity[:,1] - dmn_connectivity[:,0]
fpn_connectivity = average_within_mat[1,1,:,:] # subjects x visit
fpn_change = fpn_connectivity[:,1] - fpn_connectivity[:,0]

amyg_change = amyg[:,1] - amyg[:,0]
plt.figure()
for s in np.arange(len(DMN_subjects)):
    subjectNum = DMN_subjects[s]
    subjectInd = np.argwhere(all_subjects==subjectNum)
    if subjectNum < 100:
        style = 0
    elif subjectNum > 100:
        style = 1
    plt.plot(fpn_connectivity[s,0],amyg[subjectInd,0],marker='.',ms=20,color=colors[style])

plt.show()



plt.figure()
for s in np.arange(len(DMN_subjects)):
    subjectNum = DMN_subjects[s]
    subjectInd = np.argwhere(all_subjects==subjectNum)
    if subjectNum < 100:
        style = 0
    elif subjectNum > 100:
        style = 1
    plt.plot(fpn_change[s],amyg_change[subjectInd],marker='.',ms=20,color=colors[style])

plt.show()
g = [i for i in np.arange(len(all_subjects)) if all_subjects[i] in DMN_subjects]
x,y = nonNan(fpn_change,amyg_change[g])
scipy.stats.pearsonr(x,y)