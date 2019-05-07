import scipy.io as scio
import scipy.signal as scsig
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

folder_s = 'Ninapro-DB1'
folder_t = 'Ninapro-DB1-Proc'
subjects_num = 28
rep_index = 10
sti_index = 11
gro_index = 12
if not os.path.exists(folder_t):
    os.makedirs(folder_t)
for s in range(1,subjects_num):
	for e in range(1,4):
		file = '{}/S{}_A1_E{}.mat'.format(folder_s,s,e);
		datamat = scio.loadmat(file)
		_size = np.min([datamat['emg'].shape[0], datamat['rerepetition'].shape[0], datamat['restimulus'].shape[0]])

		data = pd.DataFrame(np.hstack((datamat['emg'][:_size,:],datamat['rerepetition'][:_size,:],datamat['restimulus'][:_size,:])))
		#Create rest repetitions
		data[rep_index].replace(to_replace=0, method='bfill', inplace=True)
		rerepetition = data[rep_index].values.reshape((-1,1))

		#Create groups
		data[gro_index] = data[sti_index].replace(to_replace=0, method='bfill')
		regroup = data[gro_index].values.reshape((-1,1))
		
		emg = data.loc[:,0:9].values
		restimulus = data[sti_index].values.reshape((-1,1))	

		not0 = np.squeeze(np.logical_not(np.isin(regroup, 0)))
		emg = emg[not0,:]
		rerepetition = rerepetition[not0]
		restimulus = restimulus[not0]
		regroup = regroup[not0]

		not0 = np.logical_not(np.isin(restimulus, 0))
		if e==2:
			restimulus[not0] += 12
			regroup += 12
		elif e==3:
			restimulus[not0] += 29
			regroup += 29

		for gesture in np.unique(restimulus):
			g_i = np.isin(restimulus, gesture)
			file = '{}/subject-{:02d}/gesture-{:02d}/rms'.format(folder_t, int(s), int(gesture))
			if not os.path.exists(file):
				os.makedirs(file)

			if gesture == 0:
				for group in np.unique(regroup):
					_g_i = np.logical_and(np.isin(regroup, group), g_i)

					for rep in np.unique(rerepetition):
						r_i = np.isin(rerepetition, rep)
						gr_i = np.squeeze(np.logical_and(_g_i, r_i))
						x = emg[gr_i,:]
						y = restimulus[gr_i]
						z = regroup[gr_i]
						w = rerepetition[gr_i]
						scio.savemat(file+'/rep-{:02d}_{:02d}.mat'.format(int(rep), int(z[0])), {'emg':x, 'stimulus':y, 'repetition':w, 'group':z})

			else:
				for rep in np.unique(rerepetition):
					r_i = np.isin(rerepetition, rep)
					gr_i = np.squeeze(np.logical_and(g_i, r_i))
					x = emg[gr_i,:]
					y = restimulus[gr_i]
					z = regroup[gr_i]
					w = rerepetition[gr_i]
					scio.savemat(file+'/rep-{:02d}.mat'.format(int(rep)), {'emg':x, 'stimulus':y, 'repetition':w, 'group':z})



