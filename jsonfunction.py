def correctjsonfiles(subjectNum,make_func_tsv):
	# purpose: make json files for bids experiment

	import os
	import glob
	from shutil import copyfile
	import pandas as pd
	import json
	import numpy as np

	bids_dir = '/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti'

	# then create empty tsv files
	ndays=3
	nifti_out="/data/jux/cnds/amennen/rtAttenPenn/fmridata/Nifti"
	bids_id = 'sub-{0:03d}'.format(subjectNum)
	for d in np.arange(ndays):
		subjectDay=d+1
		ses_id = 'ses-{0:02d}'.format(subjectDay)
		day_path=os.path.join(nifti_out,bids_id,ses_id)
		fmap_path = os.path.join(day_path,'fmap')
		fmap_fn = fmap_path + '/' + bids_id + '_' + ses_id + '_' + 'phasediff.json'
		with open(fmap_fn) as f:
			data_pd=json.load(f)
		data_pd['EchoTime1']=0.00412
		data_pd['EchoTime2']=0.00658
		func_path = os.path.join(day_path,'func')
		all_func_scans = glob.glob(os.path.join(func_path, '*.nii.gz'))
		n_func = len(all_func_scans)
		longstring = []
		#print(all_func_scans)
		for i in np.arange(n_func):
			this_run = all_func_scans[i]
			str=os.path.split(this_run)[-1]
			str_parts = str.split('_')
			full_name = ses_id + '/' + 'func' + '/' +  str_parts [0] + '_' + str_parts[1] + '_' + str_parts[2] + '_' + str_parts[3] +'_' + str_parts[4] + '_bold.nii.gz'
			#print(full_name)
			longstring.append(full_name)
		#print(longstring)
		data_pd['IntendedFor'] = longstring
		os.remove(fmap_fn)
		with open(fmap_fn, 'w') as f:
			json.dump(data_pd, f,indent=4)
		
		mag_fn = fmap_path + '/' + bids_id + '_' + ses_id + '_' + 'magnitude1.json'	
		with open(mag_fn) as f:
                        data_m1=json.load(f)
		data_m1['EchoTime1']=0.00412
		data_m1['EchoTime2']=0.00658
		data_m1['IntendedFor'] = longstring
		os.remove(mag_fn)
		with open(mag_fn, 'w') as f:
			json.dump(data_m1,f,indent=4)
		mag_fn = fmap_path + '/' + bids_id + '_' + ses_id + '_' + 'magnitude2.json'
		print(mag_fn)
		with open(mag_fn) as f:
                        data_m2=json.load(f)
		data_m2['EchoTime1']=0.00412
		data_m2['EchoTime2']=0.00658
		data_m2['IntendedFor'] = longstring
		os.remove(mag_fn)
		with open(mag_fn, 'w') as f:
			json.dump(data_m2,f,indent=4)

		# make empty .tsv files for all of the functional scans (for now)
		# read in all functional data and for each file make .tsv file
		if make_func_tsv:
			print('making functional tsv files for subject %i' % subjectNum)
			for f in np.arange(n_func):
				this_run = all_func_scans[f]
				str = os.path.split(this_run)[-1]
				str_parts = str.split('_')
				# want to keep some parts of the name
				tsvname = str_parts[0] + '_' + str_parts[1] + '_' + str_parts[2] + '_' + str_parts[3] +'_' + str_parts[4] + '_events.tsv'
				full_path = os.path.join(func_path,tsvname)
				df=pd.DataFrame(columns=['onset', 'duration', 'trial_type', 'response_time', 'stim_file'])
				df.to_csv(full_path,sep='\t',index=False)
				#with open(full_path, "w") as my_empty_csv:
				#	pass  # or write something to it already
   			
				# we're also going to want to read in that functional corrected json file and change the slice timing
				# if str_parts[3] == 'rec-corrected':
				# 	thisjson = str_parts[0] + '_' + str_parts[1] + '_' + str_parts[2] + '_' + str_parts[3] +'_' + str_parts[4] + '_bold.json'
				# 	#print(thisjson)
				# 	json_fn = os.path.join(func_path,thisjson)
				# 	with open(json_fn) as f:
				# 		data=json.load(f)
				# 	# now get slice timing we want
				# 	json_fn_uncor = os.path.join(func_path,str_parts[0] + '_' + str_parts[1] + '_' + str_parts[2] + '_rec-uncorrected_' + str_parts[4] + '_bold.json')
				# 	with open(json_fn_uncor) as f2:
				# 		data_uncor=json.load(f2)
				# 	st = data_uncor['SliceTiming']
				# 	data['SliceTiming'] = st
				# 	os.remove(json_fn)
				# 	with open(json_fn, 'w') as f:
				# 		json.dump(data,f,indent=4)
