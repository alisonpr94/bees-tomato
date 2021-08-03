import os
import csv
import sys
import pickle
import librosa
import pathlib
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Binarizer
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
import warnings
warnings.filterwarnings('ignore')

especies = ['Augochloropsis_sp1', 'Augochloropsis_sp2', 'Augochloropsis_brachycephala', 'Bombus_morio', 
		'Bombus_pauloensis', 'Centris_tarsata', 'Centris_trigonoides', 'Eulaema_nigrita', 'Exomalopsis_analis', 
		'Exomalopsis_minor', 'Melipona_bicolor', 'Melipona_quadrifasciata', 'Pseudoalglochloropsis_graminea', 
		'Xylocopa_nigrocincta', 'Xylocopa_suspecta']


def salva_modelo(modelo, nome_arq):
	filename = '/home/alison/Documentos/Projeto-Tomate/modelos/' + nome_arq
	pickle.dump(modelo, open(filename, 'wb'))

def treino_teste(matriz, classes, modelo, tuned_parameters):
	X_train, X_test, y_train, y_test = train_test_split(matriz, classes, test_size=0.5, random_state=0)
	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

	scores = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		print()
		clf = GridSearchCV(modelo, tuned_parameters, scoring=score, cv=kfold)
		
		clf.fit(X_train, y_train)

		print("Best parameters set found on development set:")
		print()
		print(clf.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r"
				% (mean, std * 2, params))
		print()

		print("Detailed classification report:")
		print()
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()
		y_true, y_pred = y_test, clf.predict(X_test)
		print("Acurácia...: %.4f" %(metrics.accuracy_score(y_true, y_pred) * 100))
		print("Precision..: %.4f" %(metrics.precision_score(y_true, y_pred, average='macro') * 100))
		print("Recall.....: %.4f" %(metrics.recall_score(y_true, y_pred, average='macro') * 100))
		print("F1-Score...: %.4f" %(metrics.f1_score(y_true, y_pred, average='macro') * 100))
		print()
		print(metrics.classification_report(y_true, y_pred))
		print()
		#print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
	fig = plt.figure(num=None, figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')
	tab_acertos = sns.heatmap(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True), 
		cmap="YlGnBu", annot=True, annot_kws={'size':14}, cbar=False, square=True)
	tab_acertos.set_xticklabels(tab_acertos.get_xticklabels(), rotation=45) 
	#tab_acertos.get_figure().savefig('heatmap.jpeg')
	plt.show()

	return clf	

def algoritmos(op, matriz, classes, zumbido):
	
	if op == "svm":	
		tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,1e-3,1e-4], 'C': [0.001, 0.1, 0.01, 1, 10], 'decision_function_shape': ['ovo']},
							{'kernel': ['poly'], 'gamma': [1e-2,1e-3,1e-4], 'C': [0.001, 0.1, 0.01, 1, 10], 'decision_function_shape': ['ovo']},
							{'kernel': ['sigmoid'], 'gamma': [1e-2,1e-3,1e-4], 'C': [0.001, 0.1, 0.01, 1, 10], 'decision_function_shape': ['ovo']}, 
							{'kernel': ['linear'], 'C': [0.001, 0.1, 0.01, 1, 10], 'decision_function_shape': ['ovo']}]

		modelo = SVC()
		clf = treino_teste(matriz, classes, modelo, tuned_parameters)
		
		modelname = "modelo_svm_mfcc_" + zumbido + ".sav"
		salva_modelo(clf, modelname)

	elif op == "lr":
		tuned_parameters = [{'penalty': ['l1', 'l2']},
							{'C': [0.001, 0.1, 1, 10, 100]}]

		modelo = LogisticRegression()
		clf = treino_teste(matriz, classes, modelo, tuned_parameters)

		modelname = "modelo_lr_mfcc_" + zumbido + ".sav"
		salva_modelo(clf, modelname)

	elif op == "dtree":
		tuned_parameters = {"criterion": ["gini", "entropy"],
							"min_samples_split": [2, 10],
							"max_depth": [2, 5, 10]
							}

		modelo = DecisionTreeClassifier()
		clf = treino_teste(matriz, classes, modelo, tuned_parameters)

		modelname = "modelo_dtree_mfcc_" + zumbido + ".sav"
		salva_modelo(clf, modelname)

	elif op == "rf":
		tuned_parameters = {'n_estimators': [100, 200],
							'max_features': ['auto', 'sqrt', 'log2']}
		
		modelo = RandomForestClassifier()
		clf = treino_teste(matriz, classes, modelo, tuned_parameters)

		modelname = "modelo_rf_mfcc_" + zumbido + ".sav"
		salva_modelo(clf, modelname)

	elif op == "ens":
		tuned_parameters = {'lr__C': [0.001, 0.1, 1, 10, 100],
							'svc__C': [0.001, 0.1, 0.01, 1, 10]}

		svc = SVC()
		rf = RandomForestClassifier()
		lr = LogisticRegression()

		modelos = [('svc', svc), ('rf', rf), ('lr', lr)]

		votingclf = VotingClassifier(estimators=modelos, voting='hard')
		clf = treino_teste(matriz, classes, votingclf, tuned_parameters)

		modelname = "modelo_ensemble_mfcc_" + zumbido + ".sav"
		salva_modelo(clf, modelname)

def read_dataset(zumbido):
	data = pd.read_csv('/home/alison/Documentos/Projeto-Tomate/datasets_especies/dataset_especies_mfcc.csv', sep=',')
	
	if zumbido == "voo":
		data = data[data['Annotation'] == 'voo']
	if zumbido == "flor":
		data = data[data['Annotation'] != 'voo']
	
	data = data.drop(['filename', 'Annotation'],axis=1)

	especies_list = data.iloc[:, -1]
	#encoder = LabelEncoder()
	#classes = encoder.fit_transform(especies_list)
	standard = StandardScaler()
	matriz = standard.fit_transform(np.array(data.iloc[:, :-1]))

	return matriz, especies_list #classes

def main(algoritmo, zumbido):

	matriz, classes = read_dataset(zumbido)
	
	algoritmos(algoritmo, matriz, classes, zumbido)

if __name__ == '__main__':

	args = sys.argv[1:]

	if len(args) >= 2:

		algoritmo = args[0]
		zumbido = args[1]
		main(algoritmo, zumbido)

	else:
		sys.exit('''
			Parâmetros incoerentes!
		''')