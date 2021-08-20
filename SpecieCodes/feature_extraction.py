import os
import csv
import pickle
import librosa
import pathlib
import pandas as pd
import numpy as np

N_MFCC = 41			 # Number of features

path_features = '/home/alison/Documentos/TomatoBeesRepository/DatasetMFCC/dataset_specie_mfcc.csv'

def feature_extraction():
	header = 'filename'
	for i in range(1, N_MFCC):
		header += f' mfcc{i}'
	
	header += ' Annotation'
	header += ' label'
	header = header.split()

	file = open(path_features, 'w', newline='')
	with file:
		writer = csv.writer(file)
		writer.writerow(header)

	species = ['Augochloropsis_sp1', 'Augochloropsis_sp2', 'Augochloropsis_brachycephala', 'Bombus_morio', 
		'Bombus_pauloensis', 'Centris_tarsata', 'Centris_trigonoides', 'Eulaema_nigrita', 'Exomalopsis_analis', 
		'Exomalopsis_minor', 'Melipona_bicolor', 'Melipona_quadrifasciata', 'Pseudoalglochloropsis_graminea', 
		'Xylocopa_nigrocincta', 'Xylocopa_suspecta']

	path_table = '/home/alison/Documentos/TomatoBeesRepository/Data/TableRecordings.xlsx'
	table = pd.read_excel(path_table)

	for s in species:
		for filename in os.listdir(f'/home/alison/Documentos/TomatoBeesRepository/Data/SpecieRecordings/{s}'):			
			songname = f'/home/alison/Documentos/TomatoBeesRepository/Data/SpecieRecordings/{s}/{filename}'
			y, sr = librosa.load(songname, mono=True)

			
			audioname = os.path.splitext(filename)[0]
			df = table[table.Audio == audioname]
			size = int(table.shape[0])

			start_time = table['Begin Time (s)']
			end_time  = table['End Time (s)']
			low_freq = table['Low Freq (Hz)']
			annotation = table['Annotation']

			for i in range(size):
				to_append = f'{filename}'
				start = float(start_time[i])
				end = float(end_time[i])
				FMIN = float(low_freq[i])
				FMAX = sr/2.0

				start_index = librosa.time_to_samples(start)
				end_index = librosa.time_to_samples(end)

				required_slice = y[start_index:end_index]

				required_mfcc = librosa.feature.mfcc(y=required_slice, sr=sr, n_mfcc=N_MFCC, fmin=FMIN, fmax=FMAX)

				for e in required_mfcc:
					to_append += f' {np.mean(e)}'
				
				to_append += f' {annotation[i]}'
				to_append += f' {s}'

				file = open(path_features, 'a', newline='')
				with file:
					writer = csv.writer(file)
					writer.writerow(to_append.split())

if __name__ == '__main__':
	extrai_features()