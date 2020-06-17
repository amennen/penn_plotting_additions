# input is config file (old or new)
# get:
# subject's name for dicom folder
# numbered ordering of all the runs
# output: subject name folder-ses01, with all the dicom folders in that structure
# PennCfg: will tell you which dicoms will go with functional scans

def transferdicoms(subject,day):
	print('transferring dicom files for subject %i, day %i' % (subject,day))

	from rtfMRI.RtfMRIClient import RtfMRIClient, loadConfigFile
	from shutil import copyfile
	import numpy as np
	import datetime
	import glob
	import os
	import shutil
	import sys

	exp_dir = "/data/jag/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo/"
	exp = exp_dir + 'subject' + str(subject) + '/usedscripts/PennCfg_Day' + str(day) + '.toml'
	#exp="/data/jag/cnds/amennen/rtAttenPenn/fmridata/behavdata/gonogo/subject1/usedscripts/PennCfg_day1.toml"
	cfg=loadConfigFile(exp)
	# add check here that subject name matches!! ###
	subjectName=cfg.session.subjectName
	subjectNum=cfg.session.subjectNum
	if subjectNum is not subject:
		raise ValueError('Subject number does not agree with cfg file')
	subjectDay=cfg.session.subjectDay
	if subjectDay is not day:
		raise ValueError('Day number does not agree with cfg file')
	if subjectDay==1:
		allRuns=np.array(cfg.session.Runs1)
		allScans=np.array(cfg.session.ScanNums1)
	elif subjectDay==2:
		allRuns=np.array(cfg.session.Runs2)
		allScans=np.array(cfg.session.ScanNums2)
	elif subjectDay==3:
		allRuns=np.array(cfg.session.Runs3)
		allScans=np.array(cfg.session.ScanNums3)
	dicom_dir="/data/jag/cnds/amennen/rtAttenPenn/fmridata/transferredImages"
	#dt = datetime.datetime.strptime(cfg.session.date,"%m/%d/%Y")
	dt = cfg.session.sessionId[0:8]
	#dicom_folder=dicom_dir + "/" +  datetime.datetime.strftime(dt,"%Y%m%d") + "." + subjectName + "." + subjectName
	dicom_folder = dicom_dir + '/' + dt + '.' + subjectName + '.' + subjectName
	print('loading from %s' % dicom_folder)
	# we know what the numbers were for the functional runs onwards
	SCAN_NUMBERS= {}
	nRuns = len(allRuns)
	for r in np.arange(nRuns):
		runNumber = r + 1
		SCAN_NUMBERS['gonogo%i' % runNumber] = allScans[r]-1 # important: both moco and non-moco versions
	#dayorder={}
	#day1order=['scout', 'T1w', 'faces1', 'faces2', 'gonogo1', 'gonogo2', 'gonogo3', 'gonogo4', 'gonogo5', 'gonogo6', 'gonogo7', 'fmap1', 'fmap2']
	allfunc={}
	allfunc[1]=['faces1', 'faces2', 'gonogo1', 'gonogo2', 'gonogo3', 'gonogo4', 'gonogo5', 'gonogo6', 'gonogo7']
	allfunc[2]=['exfunc', 'gonogo1', 'gonogo2', 'gonogo3', 'gonogo4', 'gonogo5', 'gonogo6', 'gonogo7', 'gonogo8', 'gonogo9']
	allfunc[3]=['exfunc','gonogo1', 'gonogo2', 'gonogo3', 'gonogo4', 'gonogo5', 'gonogo6', 'gonogo7', 'gonogo8','faces1', 'faces2' ]
	anat=['scout', 'T1w']
	fmap=['fmap1', 'fmap2']
	last_scan = glob.glob(dicom_folder + "/*.dcm")[-1]
	last_run = int(last_scan[-17:-11])
	DICOM_DICT = {}
	NDICOM = {}
	for s_ind in np.arange(last_run):
		this_runs_dicom = glob.glob(dicom_folder + "/001_%6.6i*.dcm" % (s_ind+1))
		DICOM_DICT[s_ind+1] = this_runs_dicom
		NDICOM[s_ind+1] = len(this_runs_dicom)
		print("FOUND: scanning run %i contains %i dicom files" %(s_ind+1,len(this_runs_dicom)))

	scout_begin_TR=128
	faces_TR=147
	gonogo_TR=242
	fmap1_TR=120
	fmap2_TR=60
	exfunc_TR=10
	anat_TR=176
	anat = [key for key,val in NDICOM.items() if val==anat_TR]
	scout1 = [key for key,val in NDICOM.items() if val==scout_begin_TR][0]
	faces = [key for key,val in NDICOM.items() if val==faces_TR]
	fmap1 = [key for key,val in NDICOM.items() if val==fmap1_TR]
	fmap2 = [key for key,val in NDICOM.items() if val==fmap2_TR]
	exfunc = [key for key,val in NDICOM.items() if val==exfunc_TR]
	SCAN_NUMBERS['scout'] = np.arange(scout1,scout1+4)
	if len(anat)>0:
		SCAN_NUMBERS['T1w'] = anat[0]
		ANAT_TAKEN = True
	else:
		ANAT_TAKEN = False
	if len(faces) > 0:
		SCAN_NUMBERS['faces1'] = faces[0]
		SCAN_NUMBERS['faces2'] = faces[2]
	if len(exfunc) > 0:
		SCAN_NUMBERS['exfunc'] = exfunc[0]
	SCAN_NUMBERS['fmap1'] = fmap1[0]
	SCAN_NUMBERS['fmap2'] = fmap2[0]
	############################################################################################
	# CHECK ALL THE SCAN NUMBERS ARE CORRECT
	############################################################################################
	print('*******************************')
	print('scan numbers are')
	for k, v in SCAN_NUMBERS.items():
		print(k, v)
	correct_scans = input('Are the scan numbers correct?y/n\n')
	if correct_scans == 'n':
		sys.exit('wrong scan numbers!\n MODIFY THEM IN %s' % exp )
	############################################################################################
	# CHECK ALL THE SCAN NUMBERS ARE CORRECT
	############################################################################################

	# now we can iterate over the whole folder and copy the dicom files over into new directory
	dicom_out="/data/jag/cnds/amennen/rtAttenPenn/fmridata/Dicom"
	bids_id = 'sub-{0:03d}'.format(subjectNum)
	ses_id = 'ses-{0:02d}'.format(subjectDay)
	day_path=os.path.join(dicom_out,bids_id,ses_id)
	# now make each directory for type of scan
	if ANAT_TAKEN:
		scantypes=['anat', 'fmap', 'func']
	else:
		scantypes=['fmap', 'func']
	for s in scantypes:
		full_path=os.path.join(day_path,s)
		if not os.path.exists(full_path):
			os.makedirs(full_path)

	# start with scout
	#src_files = DICOM_DICT[SCAN_NUMBERS['scout']]
	#dest_path = os.path.join(day_path,'anat','scout')
	#if not os.path.exists(dest_path):
	#	os.makedirs(dest_path)
	#for file_name in src_files:
	#    fn=os.path.split(file_name)[-1]
	#    dest=os.path.join(dest_path,fn)
	#    shutil.copyfile(file_name, dest)

	# first transfer anat
	if ANAT_TAKEN:
		src_files = DICOM_DICT[SCAN_NUMBERS['T1w']]
		dest_path = os.path.join(day_path,'anat', 'T1w')
		exfile=os.path.split(src_files[0])[-1]
		print('copying run from %s into %s' % (exfile,dest_path))
		if not os.path.exists(dest_path):
			os.makedirs(dest_path)
		for file_name in src_files:
			fn=os.path.split(file_name)[-1]
			dest=os.path.join(dest_path,fn)
			shutil.copyfile(file_name, dest)

	dest_path = os.path.join(day_path, 'func')
	n_func_scans = len(allfunc[subjectDay])
	scan_names=allfunc[subjectDay]
	nrunsperfunc = 2
	faces_counter=0
	gonogo_counter=0
	for f_ind in np.arange(n_func_scans):
		this_scan=scan_names[f_ind]
		if 'faces' in this_scan:
			task='faces'
			faces_counter+=1
			run=faces_counter
		elif 'gonogo' in this_scan:
			task='gonogo'
			gonogo_counter+=1
			run=gonogo_counter
		elif 'exfunc' in this_scan:
			task='exfunc'
			run=1
		
		full_name= 'task-' + task + '_' + 'rec-' + 'uncorrected' + '_' + 'run-{0:02d}'.format(run) + '_bold'
		dicom_out=os.path.join(dest_path,full_name)
		if not os.path.exists(dicom_out):
			os.makedirs(dicom_out)
		src_files = DICOM_DICT[SCAN_NUMBERS[this_scan]]
		exfile=os.path.split(src_files[0])[-1]
		print('copying run from %s into %s' % (exfile,dicom_out))
		for file_name in src_files:
			fn=os.path.split(file_name)[-1]
			dest = os.path.join(dicom_out,fn)
			shutil.copyfile(file_name, dest)
	
		full_name= 'task-' + task + '_' + 'rec-' + 'corrected' + '_' + 'run-{0:02d}'.format(run) + '_bold'
		dicom_out=os.path.join(dest_path,full_name)
		if not os.path.exists(dicom_out):
			os.makedirs(dicom_out)
		src_files = DICOM_DICT[SCAN_NUMBERS[this_scan] + 1]
		exfile=os.path.split(src_files[0])[-1]
		print('copying run from %s into %s' % (exfile,dicom_out))
		for file_name in src_files:
			fn=os.path.split(file_name)[-1]
			dest = os.path.join(dicom_out,fn)
			shutil.copyfile(file_name, dest)

	# now do same for fmap
	this_scan='fmap1'
	# first scan is magnitude
	dest_path = os.path.join(day_path,'fmap')
	full_name='magnitude1'
	dicom_out=os.path.join(dest_path,full_name)
	if not os.path.exists(dicom_out):
		os.makedirs(dicom_out)
	src_files = DICOM_DICT[SCAN_NUMBERS[this_scan]]
	exfile=os.path.split(src_files[0])[-1]
	print('copying run from %s into %s' % (exfile,dicom_out))
	for file_name in src_files:
		fn=os.path.split(file_name)[-1]
		dest = os.path.join(dicom_out,fn)
		shutil.copyfile(file_name, dest)
	
	this_scan='fmap2'
	dest_path = os.path.join(day_path,'fmap')
	full_name='phasediff'
	dicom_out=os.path.join(dest_path,full_name)
	if not os.path.exists(dicom_out):
		os.makedirs(dicom_out)
	src_files = DICOM_DICT[SCAN_NUMBERS[this_scan]]
	exfile=os.path.split(src_files[0])[-1]
	print('copying run from %s into %s' % (exfile,dicom_out))
	for file_name in src_files:
		fn=os.path.split(file_name)[-1]
		dest = os.path.join(dicom_out,fn)
		shutil.copyfile(file_name, dest)
