def makeniftis(subjectNum,subjectDay):

	from rtfMRI.RtfMRIClient import RtfMRIClient, loadConfigFile
	from shutil import copyfile
	import numpy as np
	import datetime
	import glob
	import os
	import shutil


	print('making nifti files for subject %i, day %i' % (subjectNum,subjectDay))
	dicom_out="/data/jux/cnds/amennen/rtAttenPenn/fmridata/Dicom"
	bids_id = 'sub-{0:03d}'.format(subjectNum)
	ses_id = 'ses-{0:02d}'.format(subjectDay)
	dicom_day_path=os.path.join(dicom_out,bids_id,ses_id)

	nifti_out="/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti"
	nifti_day_path=os.path.join(nifti_out,bids_id,ses_id)

	# check if there are anat files
	anat_path_dicom = os.path.join(dicom_day_path, 'anat')
	if os.path.exists(anat_path_dicom):
		T1_path_dicom = os.path.join(anat_path_dicom, 'T1w')
		this_runs_dicom = glob.glob(os.path.join(T1_path_dicom, '*'))
		if len(this_runs_dicom) == 176:
			# then high res folder exists
			T1_path_nifti = nifti_day_path + '/' + 'anat'
			if not os.path.exists(T1_path_nifti):
				os.makedirs(T1_path_nifti)
			T1_name_nifti = bids_id + '_' + ses_id + '_' + 'T1w'
			command = 'dcm2niix -f %s -o %s -z y %s' % (T1_name_nifti,T1_path_nifti,T1_path_dicom)
			os.system(command)

	# now all functional data for that day
	func_path_dicom = os.path.join(dicom_day_path, 'func')
	all_func_scans = glob.glob(os.path.join(func_path_dicom, '*'))
	n_func = len(all_func_scans)
	func_path_nifti = nifti_day_path + '/' + 'func'
	if not os.path.exists(func_path_nifti):
		os.makedirs(func_path_nifti)
	for f in np.arange(n_func):
		this_run = all_func_scans[f]
		func_name_nifti = bids_id + '_' + ses_id + '_' + os.path.split(this_run)[-1]
		command = 'dcm2niix -f %s -o %s -z y %s' % (func_name_nifti,func_path_nifti,this_run)
		os.system(command)
	# now we can iterate over the whole folder and copy the dicom files over into new directory

	# now get the fieldmap
	fmap_path_dicom=os.path.join(dicom_day_path, 'fmap')
	all_fmap_scans = glob.glob(os.path.join(fmap_path_dicom, '*'))
	n_fmap = len(all_fmap_scans)
	fmap_path_nifti = nifti_day_path + '/' + 'fmap'
	if not os.path.exists(fmap_path_nifti):
		os.makedirs(fmap_path_nifti)
	for f in np.arange(n_fmap):
		this_run = all_fmap_scans[f]
		fmap_name_nifti = bids_id + '_' + ses_id + '_' + os.path.split(this_run)[-1]
		command = 'dcm2niix -f %s -o %s -z y %s' % (fmap_name_nifti,fmap_path_nifti,this_run)
		os.system(command)
	
	# rename magnitude from _e2 to 2
	mag_name = bids_id + '_' + ses_id + '_' + 'magnitude2'
	temp_name_nifti = bids_id + '_' + ses_id + '_' + 'magnitude1'
	command = 'mv %s/%s %s/%s' % (fmap_path_nifti,temp_name_nifti + '_e2.nii.gz',fmap_path_nifti,mag_name + '.nii.gz')
	os.system(command)
	command = 'mv %s/%s %s/%s' % (fmap_path_nifti,temp_name_nifti + '_e2.json',fmap_path_nifti,mag_name + '.json')
	os.system(command)
                	 

	# it's going to label phasediff_e2
	command = 'mv %s/%s %s/%s' % (fmap_path_nifti,fmap_name_nifti + '_e2.nii.gz',fmap_path_nifti,fmap_name_nifti + '.nii.gz')
	os.system(command)
	command = 'mv %s/%s %s/%s' % (fmap_path_nifti,fmap_name_nifti + '_e2.json',fmap_path_nifti,fmap_name_nifti + '.json')
	os.system(command)

