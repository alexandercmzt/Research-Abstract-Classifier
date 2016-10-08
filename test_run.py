from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from get_unigram_features import get_unigrams
import collections
import numpy as np
import csv


def prepare_features(d2v_num, unigram_num):
	y_train = joblib.load('saves/data/train_out.csv_y_vector.pkl')
	if d2v_num != '0':
		X_train = joblib.load('saves/data/train_in.csv_feature_vectors_' + d2v_num + '.pkl')
		X_test = joblib.load('saves/data/test_in.csv_feature_vectors_' + d2v_num + '.pkl')
		if unigram_num != 0:
			UNIGRAM_train, UNIGRAM_test = get_unigrams('data/train_in.csv', 'data/test_in.csv', n=unigram_num)
			X_train = np.concatenate((X_train, UNIGRAM_train), axis=1)
			X_test = np.concatenate((X_test, UNIGRAM_test), axis=1)
			print "Full features prepared"
		else:
			X_train = X_train.tolist()
			X_test = X_test.tolist()
	else: 
		X_train, X_test = get_unigrams('data/train_in.csv', 'data/test_in.csv', n=unigram_num)
		print "Back in test run"
		return X_train, y_train, X_test
	# y_train = y_train.tolist()
	# i=0
	# while i < len(y_train):
	# 	if y_train[i] == 'category':
	# 		del X_train[i]
	# 		del y_train[i]
	# 	else:
	# 		i += 1
	# print "Bugged features removed"
	# X_train = np.array(X_train)
	# y_train = np.array(y_train)
	# X_test = np.array(X_test)
	print X_train.shape, X_test.shape, y_train.shape
	return X_train, y_train, X_test

def accuracy(gold, predict):
    assert len(gold) == len(predict)
    corr = 0
    for i in xrange(len(gold)):
        if gold[i] == predict[i]:
            corr += 1
    acc = float(corr) / len(gold)
    print 'Accuracy %d / %d = %.4f' % (corr, len(gold), acc)


def cross_validate(X_train, y_train, X_test):

	clf = svm.LinearSVC()
	boundary = len(X_train)/2
	clf.fit(X_train[:boundary], y_train[:boundary])
	accuracy(y_train[boundary:], clf.predict(X_train[boundary:]))
	# clf = LogisticRegression()
	# scores = cross_val_score(clf, X_train, y_train, cv=3)
	# print("LOGREG Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))

	# clf = svm.LinearSVC()
	# scores = cross_val_score(clf, X_train, y_train, cv=3)
	# print("SVM Accuracy (linear kernel): %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))

	# clf = AdaBoostClassifier(base_estimator=LogisticRegression())
	# scores = cross_val_score(clf, X_train, y_train, cv=3)
	# print("AdaBoost (Log Reg) Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
	
	# clf = AdaBoostClassifier()
	# scores = cross_val_score(clf, X_train, y_train, cv=3)
	# print("AdaBoost (Decision Trees) Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
	
	# clf = svm.SVC(kernel="sigmoid")
	# scores = cross_val_score(clf, X_train, y_train, cv=3)
	# print("SVM Accuracy (sigmoid kernel): %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
	
	# clf = svm.SVC()
	# scores = cross_val_score(clf, X_train, y_train, cv=3)
	# print("SVM Accuracy (rbf kernel): %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
	
	# clf = RandomForestClassifier()
	# scores = cross_val_score(clf, X_train, y_train, cv=3)
	# print("Random Forests Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
	
	# clf = ExtraTreesClassifier()
	# scores = cross_val_score(clf, X_train, y_train, cv=3)
	# print("Extremely Random Forests Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
	
	# clf = KNeighborsClassifier()
	# scores = cross_val_score(clf, X_train, y_train, cv=3)
	# print("kNN Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
	
	# clf = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
	# scores = cross_val_score(clf, X_train, y_train, cv=3)
	# print("Bagging kNN Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))


def make_output(X_train, y_train, X_test):
	clf = svm.LinearSVC()
	clf.fit(X_train,y_train)
	predictions = clf.predict(X_test)
	output = [['id', 'category']]
	for i in xrange(len(predictions)):
		output.append([i, predictions[i]])
	with open("output_55800.csv", "wb") as f:
	    writer = csv.writer(f)
	    writer.writerows(output)

def compare(a,b):
	with open(a, 'rU') as f:
		reader = csv.reader(f)
		data1 = list(list(elem) for elem in csv.reader(f, delimiter=','))
		f.close()
	with open(b, 'rU') as f:
		reader = csv.reader(f)
		data2 = list(list(elem) for elem in csv.reader(f, delimiter=','))
		f.close()
	lol = len(data1)
	correct = 0
	for i in xrange(lol):
		if data1[i] == data2[i]:
			correct+=1
	print correct
	print lol

if __name__ == "__main__":
	#compare('output_800_SVM.csv', 'output_400_SVM.csv')

	# for d2v_num in ['0']:#,'200','400', '800']:
	# 	for unigram_num in [55000, 60000, 65000]:
	# 		print "------RUN FOR D2V[" + d2v_num + "] UNIGRAM[" + str(unigram_num) + "]------"
	# 		X_train, y_train, X_test = prepare_features(d2v_num, unigram_num)
	# 		cross_validate(X_train, y_train, X_test)

	x,y,z = prepare_features('0', 55800)
	# cross_validate(x,y,z)
	make_output(x,y,z)

