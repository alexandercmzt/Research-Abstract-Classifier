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
import collections
import numpy as np
import csv

X_train = joblib.load('saves/data/train_in.csv_feature_vectors_800.pkl')
X_test = joblib.load('saves/data/test_in.csv_feature_vectors_800.pkl')
UNIGRAM_train = joblib.load('saves/data/train_in.csv_feature_vectors_UNIGRAMS.pkl')
UNIGRAM_test = joblib.load('saves/data/test_in.csv_feature_vectors_UNIGRAMS.pkl')
X_train_new = []
X_test_new = []
for i in xrange(len(X_train)):
	X_train_new.append( np.concatenate([X_train[i], UNIGRAM_train[i]]).tolist() )
for i in xrange(len(X_test)):
	X_test_new.append( np.concatenate([X_test[i], UNIGRAM_test[i]]).tolist() )
X_train = X_train_new
X_test = X_test_new
y_train = joblib.load('saves/data/train_out.csv_y_vector.pkl')
y_train = y_train.tolist()
X = []
y = []
for i in xrange(len(y_train)):
	if y_train[i] != 'category':
		X.append(X_train[i])
		y.append(y_train[i])
X_train = np.array(X)
y_train = np.array(y)
X_test = np.array(X_test)
print "Feature fixes done "
print X_train.shape, X_test.shape, y_train.shape

def accuracy(gold, predict):
    assert len(gold) == len(predict)
    corr = 0
    for i in xrange(len(gold)):
        if gold[i] == predict[i]:
            corr += 1
    acc = float(corr) / len(gold)
    print 'Accuracy %d / %d = %.4f' % (corr, len(gold), acc)

clf = LogisticRegression()
scores = cross_val_score(clf, X_train, y_train, cv=3)
print("LOGREG Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
clf = svm.LinearSVC()
scores = cross_val_score(clf, X_train, y_train, cv=3)
print("SVM Accuracy (linear kernel): %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
clf = AdaBoostClassifier(base_estimator=LogisticRegression())
scores = cross_val_score(clf, X_train, y_train, cv=3)
print("AdaBoost (Log Reg) Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
clf = AdaBoostClassifier()
scores = cross_val_score(clf, X_train, y_train, cv=3)
print("AdaBoost (Decision Trees) Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
clf = svm.SVC(kernel="sigmoid")
scores = cross_val_score(clf, X_train, y_train, cv=3)
print("SVM Accuracy (sigmoid kernel): %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
clf = svm.SVC()
scores = cross_val_score(clf, X_train, y_train, cv=3)
print("SVM Accuracy (rbf kernel): %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
clf = RandomForestClassifier()
scores = cross_val_score(clf, X_train, y_train, cv=3)
print("Random Forests Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
clf = ExtraTreesClassifier()
scores = cross_val_score(clf, X_train, y_train, cv=3)
print("Extremely Random Forests Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
clf = KNeighborsClassifier()
scores = cross_val_score(clf, X_train, y_train, cv=3)
print("kNN Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
clf = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
scores = cross_val_score(clf, X_train, y_train, cv=3)
print("Bagging kNN Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))


# partition = -1*int(len(y_train)*(0.995))
# clf = LogisticRegression()
# clf.fit(X_train[:partition], y_train[:partition])
# predictions = clf.predict(X_train[partition:])
# gold = y_train[partition:]
# accuracy(gold,predictions)

# clf = svm.LinearSVC()
# clf.fit(X_train,y_train)
# predictions = clf.predict(X_test)
# output = [['id', 'category']]
# for i in xrange(len(predictions)):
# 	output.append([i, predictions[i]])
# with open("output_1600_SVM.csv", "wb") as f:
#     writer = csv.writer(f)
#     writer.writerows(output)


# with open('output_800_SVM.csv', 'rU') as f:
# 	reader = csv.reader(f)
# 	data1 = list(list(elem) for elem in csv.reader(f, delimiter=','))
# 	f.close()
# with open('output_400_SVM.csv', 'rU') as f:
# 	reader = csv.reader(f)
# 	data2 = list(list(elem) for elem in csv.reader(f, delimiter=','))
# 	f.close()
# lol = len(data1)
# correct = 0
# for i in xrange(lol):
# 	if data1[i] == data2[i]:
# 		correct+=1

# print correct
# print lol
