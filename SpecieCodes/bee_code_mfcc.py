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

species = ['Augochloropsis_sp1', 'Augochloropsis_sp2', 'Augochloropsis_brachycephala', 'Bombus_morio', 
		'Bombus_pauloensis', 'Centris_tarsata', 'Centris_trigonoides', 'Eulaema_nigrita', 'Exomalopsis_analis', 
		'Exomalopsis_minor', 'Melipona_bicolor', 'Melipona_quadrifasciata', 'Pseudoalglochloropsis_graminea', 
		'Xylocopa_nigrocincta', 'Xylocopa_suspecta']


def save_model(model, file):
	filename = '/home/alison/Documentos/TomatoBeeRepository/models_specie/' + file
	pickle.dump(model, open(filename, 'wb'))

def train_test(matrix, classes, model, tuned_parameters):
	X_train, X_test, y_train, y_test = train_test_split(matrix, classes, test_size=0.5, random_state=0)
	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

	scores = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		print()
		clf = GridSearchCV(model, tuned_parameters, scoring=score, cv=kfold)
		
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
		print("Accuracy...: %.4f" %(metrics.accuracy_score(y_true, y_pred) * 100))
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

def algorithms(op, matrix, classes, buzz):
	
	if op == "svm":	
		tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,1e-3,1e-4], 'C': [0.001, 0.1, 0.01, 1, 10], 'decision_function_shape': ['ovo']},
							{'kernel': ['poly'], 'gamma': [1e-2,1e-3,1e-4], 'C': [0.001, 0.1, 0.01, 1, 10], 'decision_function_shape': ['ovo']},
							{'kernel': ['sigmoid'], 'gamma': [1e-2,1e-3,1e-4], 'C': [0.001, 0.1, 0.01, 1, 10], 'decision_function_shape': ['ovo']}, 
							{'kernel': ['linear'], 'C': [0.001, 0.1, 0.01, 1, 10], 'decision_function_shape': ['ovo']}]

		model = SVC()
		clf = train_test(matrix, classes, model, tuned_parameters)
		
		modelname = "model_svm_mfcc_" + buzz + ".sav"
		save_model(clf, modelname)

	elif op == "lr":
		tuned_parameters = [{'penalty': ['l1', 'l2']},
							{'C': [0.001, 0.1, 1, 10, 100]}]

		model = LogisticRegression()
		clf = train_test(matrix, classes, model, tuned_parameters)

		modelname = "model_lr_mfcc_" + buzz + ".sav"
		save_model(clf, modelname)

	elif op == "dtree":
		tuned_parameters = {"criterion": ["gini", "entropy"],
							"min_samples_split": [2, 10],
							"max_depth": [2, 5, 10]
							}

		model = DecisionTreeClassifier()
		clf = train_test(matrix, classes, model, tuned_parameters)

		modelname = "model_dtree_mfcc_" + buzz + ".sav"
		save_model(clf, modelname)

	elif op == "rf":
		tuned_parameters = {'n_estimators': [100, 200],
							'max_features': ['auto', 'sqrt', 'log2']}
		
		model = RandomForestClassifier()
		clf = train_test(matrix, classes, model, tuned_parameters)

		modelname = "model_rf_mfcc_" + buzz + ".sav"
		save_model(clf, modelname)

	elif op == "ens":
		tuned_parameters = {'lr__C': [0.001, 0.1, 1, 10, 100],
							'svc__C': [0.001, 0.1, 0.01, 1, 10]}

		svc = SVC()
		rf = RandomForestClassifier()
		lr = LogisticRegression()

		models = [('svc', svc), ('rf', rf), ('lr', lr)]

		votingclf = VotingClassifier(estimators=models, voting='hard')
		clf = train_test(matrix, classes, votingclf, tuned_parameters)

		modelname = "model_ensemble_mfcc_" + buzz + ".sav"
		save_model(clf, modelname)

def read_dataset(buzz):
	data = pd.read_csv('/home/alison/Documentos/TomatoBeeRepository/DatasetMFCC/dataset_specie_mfcc.csv', sep=',')
	
	if buzz == "flight":
		data = data[data['Annotation'] == 'flight']
	if buzz == "sonication":
		data = data[data['Annotation'] != 'flight']
	
	data = data.drop(['filename', 'Annotation'],axis=1)

	species_list = data.iloc[:, -1]
	#encoder = LabelEncoder()
	#classes = encoder.fit_transform(especies_list)
	standard = StandardScaler()
	matrix = standard.fit_transform(np.array(data.iloc[:, :-1]))

	return matrix, species_list

def main(algorithm, buzz):

	matrix, classes = read_dataset(buzz)
	
	algorithms(algorithm, matrix, classes, buzz)

if __name__ == '__main__':

	args = sys.argv[1:]

	if len(args) >= 2:

		algorithm = args[0]
		buzz = args[1]
		main(algorithm, buzz)

	else:
		sys.exit('''
			Incorrect parameters!
		''')