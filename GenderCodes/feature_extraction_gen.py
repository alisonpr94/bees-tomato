import os
import csv
import pickle
import librosa
import pathlib
import pandas as pd
import numpy as np

N_FFT = 1024         # Número de posições na frequência para Fast Fourier Transform
HOP_SIZE = 1024      # Número de quadros de áudio entre colunas STFT
SR = 44100           # Frequência de amostragem
N_MELS = 40          # Parâmetros de filtros Mel   
WIN_SIZE = 1024      # Número de amostras em cada janela STFT
WINDOW_TYPE = 'hann' # The windowin function
FEATURE = 'mel'      # Feature representation
#FMIN = 1400
N_MFCC = 41			 # Número de features


def extrai_features():
	header = 'filename'
	for i in range(1, N_MFCC):
		header += f' mfcc{i}'
	
	header += ' Peso'
	header += ' TamanhoTorax'
	header += ' Annotation'
	header += ' label'
	header = header.split()


	file = open('/home/alison/Documentos/Projeto/datasets_generos/dataset_genero_mfcc_pesoTamanho.csv', 'w', newline='')
	with file:
		writer = csv.writer(file)
		writer.writerow(header)

	genero = ['Augchloropsis', 'Bombus', 'Centris', 'Eulaema', 'Exomalopis', 'Melipona', 'Pseudoalglochloropsi', 'Xylocopa']

	for g in genero:
		for filename in os.listdir(f'/home/alison/Documentos/Projeto/gênero/{g}'):			
			songname = f'/home/alison/Documentos/Projeto/gênero/{g}/{filename}'
			y, sr = librosa.load(songname, mono=True)

			table_name = os.path.splitext(filename)[0] + ".txt"
			table = f'/home/alison/Documentos/Projeto/TabelasAudiosSeparadosGênero/{g}/{table_name}'
			table = pd.read_table(table, sep='\t')
			size = int(table.shape[0])

			start_time = table['Begin Time (s)']
			end_time  = table['End Time (s)']
			low_freq = table['Low Freq (Hz)']
			annotation = table['Annotation']
			peso_abelha = table['peso']
			tamanho_torax = table['tamanho torax']

			for i in range(size):
				to_append = f'{filename}'
				start = float(start_time[i])
				end = float(end_time[i])
				FMIN = float(low_freq[i])
				FMAX = sr/2.0
				peso = float(peso_abelha[i])
				tamanho = float(tamanho_torax[i])

				start_index = librosa.time_to_samples(start)
				end_index = librosa.time_to_samples(end)

				required_slice = y[start_index:end_index]

				required_mfcc = librosa.feature.mfcc(y=required_slice, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_SIZE, n_mels=N_MELS, htk=True, fmin=FMIN, fmax=FMAX)

				for e in required_mfcc:
					to_append += f' {np.mean(e)}'
				
				to_append += f' {peso}'
				to_append += f' {tamanho}'
				to_append += f' {annotation[i]}'
				to_append += f' {g}'

				file = open('/home/alison/Documentos/Projeto/datasets_generos/dataset_genero_mfcc_pesoTamanho.csv', 'a', newline='')
				with file:
					writer = csv.writer(file)
					writer.writerow(to_append.split())

if __name__ == '__main__':
	extrai_features()